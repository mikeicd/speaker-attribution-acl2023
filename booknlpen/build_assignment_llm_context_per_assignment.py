import argparse
import ast
import json
import os
from datetime import datetime, timezone

import pandas as pd

from booknlpen import create_book_corefs


DEFAULT_CONTEXT_WINDOW = 40


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


def _make_context_getter(tokens, window):
    tokens = tokens.sort_values('token_ID_within_document')
    token_ids = [int(token_id) for token_id in tokens['token_ID_within_document']]
    words = [str(word) for word in tokens['word']]
    positions = {token_id: index for index, token_id in enumerate(token_ids)}

    def get_context(start_token, end_token, marker=None):
        if start_token not in positions or end_token not in positions:
            return {
                'available': False,
                'before': None,
                'span': None,
                'after': None,
                'text': None,
            }

        start = positions[start_token]
        end = positions[end_token]
        if end < start:
            return {
                'available': False,
                'before': None,
                'span': None,
                'after': None,
                'text': None,
            }

        before = ' '.join(words[max(0, start - window):start])
        span = ' '.join(words[start:end + 1])
        after = ' '.join(words[end + 1:end + 1 + window])
        marked_span = '{}{}{}'.format(marker[0], span, marker[1]) if marker else span
        text = ' '.join(part for part in (before, marked_span, after) if part)
        return {
            'available': True,
            'before': before,
            'span': span,
            'after': after,
            'text': text,
        }

    return get_context


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


def build_novel_assignment_context(novel, context_window=DEFAULT_CONTEXT_WINDOW):
    """Build the original LLM package with one context entry per assignment."""
    if novel not in create_book_corefs.novels:
        raise ValueError("Unknown novel: {}".format(novel))
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

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
    tokens = create_book_corefs.read_booknlp_df(tokens_path)
    matches = pd.read_csv(matches_path)
    char_info = create_book_corefs.read_char_info(novel)
    get_context = _make_context_getter(tokens, context_window)
    match_by_coref = {
        int(row.COREF): row for row in matches.itertuples(index=False)
    }

    assignments_by_cluster = {}
    for quote_index, row in enumerate(quotes.itertuples(index=False)):
        coref_id = _as_int(row.char_id)
        mention_start = _as_int(row.mention_start)
        mention_end = _as_int(row.mention_end)
        if coref_id is None or mention_start is None or mention_end is None:
            continue

        status, candidate_ids = _cluster_resolution(coref_id, match_by_coref)
        if status == 'resolved':
            continue

        quote_start = int(row.quote_start)
        quote_end = int(row.quote_end)
        assignment = {
            'assignment_id': '{}:quote:{}'.format(novel, quote_index),
            'quote_index': quote_index,
            'quote_start_token': quote_start,
            'quote_end_token': quote_end,
            'quote_text': str(row.quote),
            'quote_context': get_context(
                quote_start, quote_end, marker=('[[', ']]')
            ),
            'selected_mention_text': str(row.mention_phrase),
            'selected_mention_start_token': mention_start,
            'selected_mention_end_token': mention_end,
            'selected_mention_context': get_context(
                mention_start, mention_end, marker=('[[', ']]')
            ),
        }
        cluster = assignments_by_cluster.setdefault(coref_id, {
            'COREF': coref_id,
            'cluster_key': '{}:{}'.format(novel, coref_id),
            'resolution_status': status,
            'existing_candidate_pdnc_ids': candidate_ids,
            'assignments': [],
        })
        cluster['assignments'].append(assignment)

    entities_by_cluster = {
        int(coref_id): group
        for coref_id, group in entities.groupby('COREF', sort=False)
    }
    for coref_id, cluster in assignments_by_cluster.items():
        mention_rows = []
        entity_group = entities_by_cluster.get(coref_id)
        if entity_group is not None:
            for row in entity_group.sort_values(
                'start_token'
            ).itertuples(index=False):
                start = int(row.start_token)
                end = int(row.end_token)
                mention_rows.append({
                    'text': str(row.text),
                    'prop': str(row.prop),
                    'category': str(row.cat),
                    'start_token': start,
                    'end_token': end,
                    'context': get_context(start, end, marker=('[[', ']]')),
                })

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
        cluster['assignment_count'] = len(cluster['assignments'])
        cluster['mention_count'] = len(mention_rows)
        cluster['categories'] = sorted({
            mention['category'] for mention in mention_rows
        })
        cluster['mentions'] = mention_rows

    character_catalog = _character_catalog(char_info)
    clusters = sorted(
        assignments_by_cluster.values(),
        key=lambda cluster: cluster['COREF'],
    )
    package = {
        'schema_version': 1,
        'packaging_strategy': 'per_assignment',
        'novel': novel,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': (
            'Original package format: one quote and selected-mention context '
            'entry per unresolved assignment.'
        ),
        'candidate_policy': {
            'eligible_clusters': (
                'Unresolved or unmatched clusters selected as speakers in '
                'BookNLP quote assignments.'
            ),
            'allowed_character_ids': [
                character['pdncID'] for character in character_catalog
            ],
            'abstention_allowed': True,
            'resolution_status_source': 'coref_matches.csv',
        },
        'context_window_tokens_per_side': context_window,
        'source_files': {
            'quotes': quotes_path,
            'entities': entities_path,
            'tokens': tokens_path,
            'coref_matches': matches_path,
        },
        'character_catalog': character_catalog,
        'total_quote_count': len(quotes),
        'assignment_count': sum(
            cluster['assignment_count'] for cluster in clusters
        ),
        'unresolved_assignment_rate': (
            sum(cluster['assignment_count'] for cluster in clusters) / len(quotes)
            if len(quotes) else 0.0
        ),
        'unresolved_assignment_cluster_count': len(clusters),
        'clusters': clusters,
        'llm_response_schema': {
            'novel': novel,
            'decisions': [
                {
                    'cluster_key': '{}:<COREF>'.format(novel),
                    'COREF': '<integer from the package>',
                    'decision': 'resolve or abstain',
                    'pdncID': '<allowed character id, or null for abstain>',
                    'confidence': '<optional number from 0 to 1>',
                    'evidence_assignment_ids': [
                        '<assignment_id values from the package>'
                    ],
                    'rationale': '<brief explanation>',
                }
            ],
        },
    }

    output_path = os.path.join(
        novel_output, 'speaker_assignment_llm_context_packages_per_assignment.json'
    )
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(package, output_file, indent=2, ensure_ascii=False)

    summary = {
        'novel': novel,
        'packaging_strategy': 'per_assignment',
        'total_quote_assignments': len(quotes),
        'unresolved_assignment_count': package['assignment_count'],
        'unresolved_assignment_rate': package['unresolved_assignment_rate'],
        'unresolved_assignment_cluster_count': len(clusters),
        'assignment_context_entry_count': package['assignment_count'],
        'expected_cluster_decision_count': len(clusters),
        'context_package': output_path,
    }
    summary_path = os.path.join(
        novel_output, 'speaker_assignment_llm_context_summary_per_assignment.json'
    )
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print(
        '{}: {} assignment-context entries in {} unresolved clusters; wrote {}'.format(
            novel,
            package['assignment_count'],
            len(clusters),
            output_path,
        )
    )
    return package


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build the original one-context-per-assignment LLM evidence packages.'
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--novel', choices=create_book_corefs.novels)
    selection.add_argument(
        '--all-novels',
        action='store_true',
        help='Build packages for all PDNC novels with restored checkpoints.',
    )
    parser.add_argument(
        '--context-window',
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help='Number of surrounding tokens on each side of a span.',
    )
    args = parser.parse_args()
    novels = create_book_corefs.novels if args.all_novels else [args.novel]
    for novel in novels:
        build_novel_assignment_context(novel, args.context_window)


if __name__ == '__main__':
    main()
