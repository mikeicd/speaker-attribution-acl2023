import argparse
import ast
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from booknlpen import create_book_corefs
from booknlpen.assignment_llm_artifacts import (
    context_package_filename,
    context_summary_filename,
    split_passage_group,
)


DEFAULT_CONTEXT_WINDOW = 40
DEFAULT_MAX_PASSAGE_TOKENS = 1200


def _as_int(value):
    if pd.isna(value):
        return None
    return int(value)


def _parse_match_ids(value):
    if isinstance(value, list):
        raw_ids = value
    elif pd.isna(value):
        raw_ids = []
    else:
        raw_ids = ast.literal_eval(str(value))

    if not isinstance(raw_ids, list):
        raise ValueError("Expected coref_matches.matches to contain a list")
    return list(dict.fromkeys(int(candidate_id) for candidate_id in raw_ids))


def _group_overlapping_ranges(rows, max_passage_tokens):
    rows = sorted(rows, key=lambda row: (row['context_start'], row['context_end']))
    groups = []
    current = []
    current_start = None
    current_end = None

    for row in rows:
        start = row['context_start']
        end = row['context_end']
        overlaps = current and start <= current_end + 1
        within_limit = (
            overlaps
            and max(current_end, end) - current_start + 1 <= max_passage_tokens
        )
        if current and not within_limit:
            groups.append({
                'rows': current,
                'context_start': current_start,
                'context_end': current_end,
            })
            current = []
            current_start = None
            current_end = None

        if not current:
            current_start = start
            current_end = end
        else:
            current_end = max(current_end, end)
        current.append(row)

    if current:
        groups.append({
            'rows': current,
            'context_start': current_start,
            'context_end': current_end,
        })
    return groups


def _annotated_passage_text(words, start, end, assignments):
    events = defaultdict(list)
    for assignment in assignments:
        assignment_id = assignment['assignment_id']
        quote_start = max(assignment['quote_start_position'], start) - start
        quote_end = min(assignment['quote_end_position'], end) - start + 1
        events[quote_start].append(
            (1, '[[QUOTE_START:{}]]'.format(assignment_id))
        )
        events[quote_end].append(
            (0, '[[QUOTE_END:{}]]'.format(assignment_id))
        )

        mention_start = assignment['selected_mention_start_position']
        mention_end = assignment['selected_mention_end_position']
        if (
            mention_start is not None
            and mention_end is not None
            and mention_start <= end
            and mention_end >= start
        ):
            events[max(mention_start, start) - start].append(
                (1, '[[MENTION_START:{}]]'.format(assignment_id))
            )
            events[min(mention_end, end) - start + 1].append(
                (0, '[[MENTION_END:{}]]'.format(assignment_id))
            )

    parts = []
    for offset, word in enumerate(words[start:end + 1]):
        parts.extend(marker for _, marker in sorted(events[offset]))
        parts.append(word)
    parts.extend(marker for _, marker in sorted(events[end - start + 1]))
    return ' '.join(parts)


def _character_catalog(char_info):
    catalog = []
    for raw_id, raw_names in sorted(
        char_info['id2names'].items(), key=lambda item: int(item[0])
    ):
        character_id = int(raw_id)
        names = raw_names if isinstance(raw_names, list) else [raw_names]
        catalog.append({
            'pdncID': character_id,
            'canonical_name': str(char_info['id2parent'].get(character_id, '')),
            'names': [str(name) for name in names],
        })
    return catalog


def _cluster_resolution(coref_id, match_by_coref):
    row = match_by_coref.get(coref_id)
    if row is None:
        return 'missing_cluster_match', []

    candidate_ids = _parse_match_ids(row.matches)
    pdnc_id = _as_int(row.pdncID)
    if pdnc_id is not None and pdnc_id >= 0:
        return 'resolved', candidate_ids
    if len(candidate_ids) > 1:
        return 'ambiguous', candidate_ids
    if len(candidate_ids) == 1:
        return 'unresolved_single_candidate', candidate_ids
    return 'unmatched', candidate_ids


def build_novel_assignment_context(
    novel,
    context_window=DEFAULT_CONTEXT_WINDOW,
    max_passage_tokens=DEFAULT_MAX_PASSAGE_TOKENS,
    max_decisions_per_passage=None,
):
    """Build shared, bounded LLM passages for unresolved speaker assignments."""
    if novel not in create_book_corefs.novels:
        raise ValueError("Unknown novel: {}".format(novel))
    if context_window < 0:
        raise ValueError("context_window must be non-negative")
    if max_passage_tokens < 1:
        raise ValueError("max_passage_tokens must be positive")
    if (
        max_decisions_per_passage is not None
        and max_decisions_per_passage < 1
    ):
        raise ValueError("max_decisions_per_passage must be positive")

    novel_output = os.path.join(create_book_corefs.COREF_ROOT, novel)
    quotes_path = os.path.join(novel_output, novel + '.quotes')
    entities_path = os.path.join(novel_output, novel + '.entities')
    tokens_path = os.path.join(novel_output, novel + '.tokens')
    matches_path = os.path.join(novel_output, 'coref_matches.csv')
    required_paths = [quotes_path, entities_path, tokens_path, matches_path]
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            "Missing BookNLP checkpoint inputs: {}".format(', '.join(missing_paths))
        )

    quotes = create_book_corefs.read_booknlp_df(quotes_path)
    entities = create_book_corefs.read_booknlp_df(entities_path)
    tokens = create_book_corefs.read_booknlp_df(tokens_path).sort_values(
        'token_ID_within_document'
    )
    matches = pd.read_csv(matches_path)
    char_info = create_book_corefs.read_char_info(novel)
    token_ids = [int(value) for value in tokens['token_ID_within_document']]
    words = [str(value) for value in tokens['word']]
    paragraph_ids = [int(value) for value in tokens['paragraph_ID']]
    token_positions = {token_id: position for position, token_id in enumerate(token_ids)}
    match_by_coref = {
        int(row.COREF): row for row in matches.itertuples(index=False)
    }
    mentions_by_paragraph = defaultdict(list)
    for row in entities.itertuples(index=False):
        start_token = int(row.start_token)
        end_token = int(row.end_token)
        start_position = token_positions.get(start_token)
        end_position = token_positions.get(end_token)
        if start_position is None or end_position is None:
            raise ValueError(
                'Entity mention references token IDs absent from {}'.format(
                    tokens_path
                )
            )
        mentions_by_paragraph[paragraph_ids[start_position]].append({
            'COREF': int(row.COREF),
            'text': str(row.text),
            'prop': str(row.prop),
            'category': str(row.cat),
            'start_token': start_token,
            'end_token': end_token,
            'start_position': start_position,
            'end_position': end_position,
        })

    assignment_rows = []
    assignments_by_paragraph = defaultdict(list)
    unresolved_context_rows = []
    for quote_index, row in enumerate(quotes.itertuples(index=False)):
        coref_id = _as_int(row.char_id)
        mention_start = _as_int(row.mention_start)
        mention_end = _as_int(row.mention_end)
        quote_start = int(row.quote_start)
        quote_end = int(row.quote_end)
        quote_start_position = token_positions.get(quote_start)
        quote_end_position = token_positions.get(quote_end)
        if quote_start_position is None or quote_end_position is None:
            raise ValueError(
                'Quote {} references token IDs absent from {}'.format(
                    quote_index, tokens_path
                )
            )

        if coref_id is None or mention_start is None or mention_end is None:
            status, candidate_ids = 'no_attributed_mention', []
            pdnc_id = None
            resolved_character_name = None
            mention_start_position = None
            mention_end_position = None
        else:
            status, candidate_ids = _cluster_resolution(coref_id, match_by_coref)
            match_row = match_by_coref.get(coref_id)
            pdnc_id = _as_int(match_row.pdncID) if match_row is not None else None
            resolved_character_name = None
            if status == 'resolved' and pdnc_id is not None:
                resolved_character_name = str(
                    char_info['id2parent'].get(
                        pdnc_id,
                        char_info['id2names'].get(pdnc_id, ''),
                    )
                )
            mention_start_position = token_positions.get(mention_start)
            mention_end_position = token_positions.get(mention_end)

        paragraph_id = paragraph_ids[quote_start_position]
        assignment = {
            'assignment_id': '{}:quote:{}'.format(novel, quote_index),
            'quote_index': quote_index,
            'COREF': coref_id,
            'cluster_key': (
                '{}:{}'.format(novel, coref_id) if coref_id is not None else None
            ),
            'resolution_status': status,
            'decision_required': status not in ('resolved', 'no_attributed_mention'),
            'existing_candidate_pdnc_ids': candidate_ids,
            'resolved_pdnc_id': pdnc_id if status == 'resolved' else None,
            'resolved_character_name': resolved_character_name,
            'quote_paragraph_id': paragraph_id,
            'quote_start_token': quote_start,
            'quote_end_token': quote_end,
            'quote_text': str(row.quote),
            'quote_start_position': quote_start_position,
            'quote_end_position': quote_end_position,
            'selected_mention_text': (
                str(row.mention_phrase) if mention_start is not None else None
            ),
            'selected_mention_start_token': mention_start,
            'selected_mention_end_token': mention_end,
            'selected_mention_start_position': mention_start_position,
            'selected_mention_end_position': mention_end_position,
        }
        assignment_rows.append(assignment)
        assignments_by_paragraph[paragraph_id].append(assignment)
        if assignment['decision_required']:
            assignment['context_start'] = max(
                0, quote_start_position - context_window
            )
            assignment['context_end'] = min(
                len(token_ids) - 1, quote_end_position + context_window
            )
            unresolved_context_rows.append(assignment)

    character_catalog = _character_catalog(char_info)
    passages = []
    all_unresolved_clusters = {
        assignment['COREF'] for assignment in assignment_rows
        if assignment['decision_required'] and assignment['COREF'] is not None
    }
    grouped_ranges = _group_overlapping_ranges(
        unresolved_context_rows, max_passage_tokens
    )
    passage_groups = [
        group
        for original_group in grouped_ranges
        for group in split_passage_group(
            original_group, max_decisions_per_passage
        )
    ]
    for group in passage_groups:
        start = group['context_start']
        end = group['context_end']
        paragraph_id = paragraph_ids[start]
        end_paragraph_id = paragraph_ids[end]
        decision_ids = {row['assignment_id'] for row in group['rows']}
        candidate_assignments = []
        for candidate_paragraph_id in sorted({
            value for value in paragraph_ids[start:end + 1]
        }):
            candidate_assignments.extend(
                assignments_by_paragraph.get(candidate_paragraph_id, ())
            )

        local_assignments = []
        mention_contexts = {}
        for assignment in candidate_assignments:
            if (
                assignment['quote_start_position'] > end
                or assignment['quote_end_position'] < start
            ):
                continue
            passage_assignment = dict(assignment)
            passage_assignment['decision_required'] = (
                assignment['assignment_id'] in decision_ids
            )
            quote_span_start = max(assignment['quote_start_position'], start)
            quote_span_end = min(assignment['quote_end_position'], end)
            passage_assignment['quote_span_in_passage'] = [
                quote_span_start - start,
                quote_span_end - start,
            ]
            passage_assignment['quote_span_clipped'] = (
                quote_span_start != assignment['quote_start_position']
                or quote_span_end != assignment['quote_end_position']
            )
            mention_start = assignment['selected_mention_start_position']
            mention_end = assignment['selected_mention_end_position']
            if (
                mention_start is None
                or mention_end is None
                or mention_start > end
                or mention_end < start
            ):
                passage_assignment['selected_mention_span_in_passage'] = None
                passage_assignment['selected_mention_span_clipped'] = False
                passage_assignment['selected_mention_context_id'] = None
            else:
                mention_span_start = max(mention_start, start)
                mention_span_end = min(mention_end, end)
                passage_assignment['selected_mention_span_in_passage'] = [
                    mention_span_start - start,
                    mention_span_end - start,
                ]
                passage_assignment['selected_mention_span_clipped'] = (
                    mention_span_start != mention_start
                    or mention_span_end != mention_end
                )
                mention_context_id = None
                if mention_start < start or mention_end > end:
                    mention_context_id = '{}:mention:{}-{}'.format(
                        novel,
                        assignment['selected_mention_start_token'],
                        assignment['selected_mention_end_token'],
                    )
                    mention_contexts.setdefault(mention_context_id, {
                        'mention_context_id': mention_context_id,
                        'token_start': token_ids[
                            max(0, mention_start - context_window)
                        ],
                        'token_end': token_ids[
                            min(len(token_ids) - 1, mention_end + context_window)
                        ],
                        'text': ' '.join(
                            words[
                                max(0, mention_start - context_window):mention_start
                            ]
                            + [
                                '[[MENTION_START:{}]]'.format(mention_context_id)
                            ]
                            + words[mention_start:mention_end + 1]
                            + [
                                '[[MENTION_END:{}]]'.format(mention_context_id)
                            ]
                            + words[
                                mention_end + 1:
                                min(len(words), mention_end + context_window + 1)
                            ]
                        ),
                    })
                passage_assignment['selected_mention_context_id'] = (
                    mention_context_id
                )
            local_assignments.append(passage_assignment)

        passage_clusters = {}
        for assignment in local_assignments:
            coref_id = assignment['COREF']
            if coref_id is None:
                continue
            cluster = passage_clusters.setdefault(coref_id, {
                'COREF': coref_id,
                'cluster_key': assignment['cluster_key'],
                'resolution_status': assignment['resolution_status'],
                'existing_candidate_pdnc_ids': (
                    assignment['existing_candidate_pdnc_ids']
                ),
                'resolved_pdnc_id': assignment['resolved_pdnc_id'],
                'resolved_character_name': assignment['resolved_character_name'],
                'assignment_ids': [],
                'decision_assignment_ids': [],
            })
            cluster['assignment_ids'].append(assignment['assignment_id'])
            if assignment['decision_required']:
                cluster['decision_assignment_ids'].append(
                    assignment['assignment_id']
                )

        for cluster in passage_clusters.values():
            candidate_names = []
            for candidate_id in cluster['existing_candidate_pdnc_ids']:
                names = char_info['id2names'].get(candidate_id, [])
                if not isinstance(names, list):
                    names = [names]
                candidate_names.append({
                    'pdncID': candidate_id,
                    'names': [str(name) for name in names],
                })
            cluster['existing_candidate_characters'] = candidate_names
            cluster['local_mentions'] = []

        local_mentions = []
        for candidate_paragraph_id in sorted({
            value for value in paragraph_ids[start:end + 1]
        }):
            local_mentions.extend(
                mentions_by_paragraph.get(candidate_paragraph_id, ())
            )
        for mention in local_mentions:
            cluster = passage_clusters.get(mention['COREF'])
            if (
                cluster is None
                or mention['start_position'] > end
                or mention['end_position'] < start
            ):
                continue
            mention_start = max(mention['start_position'], start)
            mention_end = min(mention['end_position'], end)
            cluster['local_mentions'].append({
                'text': mention['text'],
                'prop': mention['prop'],
                'category': mention['category'],
                'start_token': mention['start_token'],
                'end_token': mention['end_token'],
                'span_in_passage': [
                    mention_start - start,
                    mention_end - start,
                ],
                'span_clipped': (
                    mention_start != mention['start_position']
                    or mention_end != mention['end_position']
                ),
            })

        passage_id = '{}:p{}-{}:t{}-{}'.format(
            novel,
            paragraph_id,
            end_paragraph_id,
            token_ids[start],
            token_ids[end],
        )
        if 'part_number' in group:
            passage_id = '{}:part{}-of-{}'.format(
                passage_id, group['part_number'], group['part_count']
            )
        passage = {
            'passage_id': passage_id,
            'paragraph_id': paragraph_id,
            'end_paragraph_id': end_paragraph_id,
            'token_start': token_ids[start],
            'token_end': token_ids[end],
            'token_count': end - start + 1,
            'text': _annotated_passage_text(
                words, start, end, local_assignments
            ),
            'selected_mention_contexts': list(mention_contexts.values()),
            'expected_decision_count': len(decision_ids),
            'assignments': [
                {
                    key: value
                    for key, value in assignment.items()
                    if key not in {
                        'quote_start_position',
                        'quote_end_position',
                        'quote_text',
                        'selected_mention_start_position',
                        'selected_mention_end_position',
                        'context_start',
                        'context_end',
                        'paragraph_id',
                    }
                }
                for assignment in local_assignments
            ],
            'speaker_clusters': sorted(
                passage_clusters.values(), key=lambda cluster: cluster['COREF']
            ),
        }
        if 'part_number' in group:
            passage['decision_part'] = {
                'number': group['part_number'],
                'count': group['part_count'],
            }
        passages.append(passage)

    unresolved_assignment_count = sum(
        assignment['decision_required'] for assignment in assignment_rows
    )
    passage_assignment_entry_count = sum(
        len(passage['assignments']) for passage in passages
    )
    passage_decision_count = sum(
        passage['expected_decision_count'] for passage in passages
    )
    multi_assignment_passage_count = sum(
        passage['expected_decision_count'] > 1 for passage in passages
    )
    multi_cluster_passage_count = sum(
        sum(bool(cluster['decision_assignment_ids'])
            for cluster in passage['speaker_clusters']) > 1
        for passage in passages
    )
    package = {
        'schema_version': 2,
        'novel': novel,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': (
            'Infer speakers for unresolved BookNLP quote assignments from shared '
            'passage context while preserving quote and cluster identities.'
        ),
        'candidate_policy': {
            'eligible_clusters': (
                'Unresolved or unmatched clusters selected as speakers in '
                'BookNLP quote assignments.'
            ),
            'resolved_cluster_assignments_are_context_only': True,
            'decision_granularity': 'one decision per unresolved quote assignment',
            'allowed_character_ids': [
                character['pdncID'] for character in character_catalog
            ],
            'abstention_allowed': True,
            'resolution_status_source': 'coref_matches.csv',
        },
        'context_window_tokens_per_side': context_window,
        'max_passage_tokens': max_passage_tokens,
        'max_decisions_per_passage': max_decisions_per_passage,
        'passage_limit_is_soft_for_single_assignment_context': True,
        'source_files': {
            'quotes': quotes_path,
            'entities': entities_path,
            'tokens': tokens_path,
            'coref_matches': matches_path,
        },
        'character_catalog': character_catalog,
        'total_quote_count': len(quotes),
        'assignment_count': unresolved_assignment_count,
        'unresolved_assignment_rate': (
            unresolved_assignment_count / len(quotes)
            if len(quotes) else 0.0
        ),
        'unresolved_assignment_cluster_count': len(all_unresolved_clusters),
        'passage_count': len(passages),
        'expected_llm_call_count': len(passages),
        'expected_output_decision_count': passage_decision_count,
        'passage_assignment_entry_count': passage_assignment_entry_count,
        'multi_assignment_passage_count': multi_assignment_passage_count,
        'multi_cluster_passage_count': multi_cluster_passage_count,
        'max_observed_passage_tokens': max(
            (passage['token_count'] for passage in passages), default=0
        ),
        'passages_over_merge_limit_count': sum(
            passage['token_count'] > max_passage_tokens for passage in passages
        ),
        'unique_input_assignment_count': len({
            assignment['assignment_id'] for passage in passages
            for assignment in passage['assignments']
        }),
        'unique_decision_assignment_count': len({
            assignment['assignment_id'] for passage in passages
            for assignment in passage['assignments']
            if assignment['decision_required']
        }),
        'mean_assignments_per_passage': (
            passage_assignment_entry_count / len(passages) if passages else 0
        ),
        'call_reduction_vs_one_call_per_unresolved_assignment': (
            1 - len(passages) / unresolved_assignment_count
            if unresolved_assignment_count else 0
        ),
        'passages': passages,
        'llm_response_schema': {
            'novel': novel,
            'passage_id': '<passage_id from the package>',
            'decisions': [
                {
                    'assignment_id': '{}:quote:<quote_index>'.format(novel),
                    'cluster_key': '{}:<COREF>'.format(novel),
                    'COREF': '<integer from the package, or null>',
                    'decision': 'resolve or abstain',
                    'pdncID': '<allowed character id, or null for abstain>',
                    'confidence': '<optional number from 0 to 1>',
                    'rationale': '<brief explanation>',
                }
            ],
        },
    }

    if (
        passage_decision_count != unresolved_assignment_count
        or package['unique_decision_assignment_count']
        != unresolved_assignment_count
    ):
        raise ValueError(
            'Decision assignments are missing or duplicated across passages'
        )
    if max_decisions_per_passage is not None and any(
        passage['expected_decision_count'] > max_decisions_per_passage
        for passage in passages
    ):
        raise ValueError('A passage exceeds max_decisions_per_passage')

    output_path = os.path.join(
        novel_output, context_package_filename(max_decisions_per_passage)
    )
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(package, output_file, indent=2, ensure_ascii=False)

    summary = {
        'novel': novel,
        'total_quote_assignments': len(quotes),
        'unresolved_assignment_count': package['assignment_count'],
        'unresolved_assignment_rate': package['unresolved_assignment_rate'],
        'unresolved_assignment_cluster_count': (
            package['unresolved_assignment_cluster_count']
        ),
        'passage_count': package['passage_count'],
        'expected_llm_call_count': package['expected_llm_call_count'],
        'expected_output_decision_count': package['expected_output_decision_count'],
        'call_reduction_vs_one_call_per_unresolved_assignment': (
            package['call_reduction_vs_one_call_per_unresolved_assignment']
        ),
        'passage_assignment_entry_count': package['passage_assignment_entry_count'],
        'multi_assignment_passage_count': package['multi_assignment_passage_count'],
        'multi_cluster_passage_count': package['multi_cluster_passage_count'],
        'max_observed_passage_tokens': package['max_observed_passage_tokens'],
        'passages_over_merge_limit_count': package[
            'passages_over_merge_limit_count'
        ],
        'max_decisions_per_passage': max_decisions_per_passage,
        'context_package': output_path,
    }
    summary_path = os.path.join(
        novel_output, context_summary_filename(max_decisions_per_passage)
    )
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print(
        '{}: {} unresolved assignments -> {} passages/{} calls; {} decisions '
        'expected ({:.1%} fewer calls than one per assignment); wrote {}'.format(
            novel,
            package['assignment_count'],
            package['passage_count'],
            package['expected_llm_call_count'],
            package['expected_output_decision_count'],
            package['call_reduction_vs_one_call_per_unresolved_assignment'],
            output_path,
        )
    )
    return package


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build LLM evidence packages for unresolved BookNLP speaker assignments.'
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--novel', choices=create_book_corefs.novels)
    selection.add_argument(
        '--all-novels',
        action='store_true',
        help='Build packages for all novels with restored BookNLP checkpoints.',
    )
    parser.add_argument(
        '--context-window',
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help='Number of surrounding tokens on each side of a span.',
    )
    parser.add_argument(
        '--max-passage-tokens',
        type=int,
        default=DEFAULT_MAX_PASSAGE_TOKENS,
        help='Maximum token count when merging overlapping quote contexts.',
    )
    parser.add_argument(
        '--max-decisions-per-passage',
        type=int,
        help=(
            'Split decision targets into passage parts with at most this many '
            'assignments. Context remains shared across the parts.'
        ),
    )
    args = parser.parse_args()
    novels = create_book_corefs.novels if args.all_novels else [args.novel]
    for novel in novels:
        build_novel_assignment_context(
            novel,
            args.context_window,
            args.max_passage_tokens,
            args.max_decisions_per_passage,
        )


if __name__ == '__main__':
    main()
