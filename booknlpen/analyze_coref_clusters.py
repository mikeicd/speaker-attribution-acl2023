import argparse
import ast
import json
import os
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from booknlpen import create_book_corefs


def _parse_match_ids(value):
    if isinstance(value, list):
        raw_ids = value
    elif pd.isna(value):
        raw_ids = []
    else:
        raw_ids = ast.literal_eval(str(value))

    if not isinstance(raw_ids, list):
        raise ValueError("Expected the 'matches' field to contain a list")

    return list(dict.fromkeys(int(match_id) for match_id in raw_ids))


def _mention_context(start_token, end_token, token_texts, token_positions, window):
    if start_token not in token_positions or end_token not in token_positions:
        return "", False

    start_pos = token_positions[start_token]
    end_pos = token_positions[end_token]
    if end_pos < start_pos:
        return "", False

    mention = " ".join(token_texts[start_pos:end_pos + 1])
    before = token_texts[max(0, start_pos - window):start_pos]
    after = token_texts[end_pos + 1:end_pos + 1 + window]
    context = " ".join(before + ["[[" + mention + "]]"] + after)
    return context, True


def generate_coref_resolution_report(novel, context_window=40, refresh_matches=False):
    """Write cluster-level resolution statistics and review evidence for one novel."""
    if novel not in create_book_corefs.novels:
        raise ValueError("Unknown novel: {}".format(novel))
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    novel_output = os.path.join(create_book_corefs.SAVE_ROOT, novel)
    os.makedirs(novel_output, exist_ok=True)
    matches_path = os.path.join(novel_output, 'coref_matches.csv')
    if refresh_matches or not os.path.isfile(matches_path):
        create_book_corefs.booknlp_process_novel(novel)

    entities_path = os.path.join(
        create_book_corefs.COREF_ROOT, novel, novel + '.entities'
    )
    tokens_path = os.path.join(
        create_book_corefs.COREF_ROOT, novel, novel + '.tokens'
    )
    entities = create_book_corefs.read_booknlp_df(entities_path)
    tokens = create_book_corefs.read_booknlp_df(tokens_path)
    matches = pd.read_csv(matches_path)
    char_info = create_book_corefs.read_char_info(novel)
    paragraph_path = os.path.join(
        create_book_corefs.DATA_ROOT, novel, 'para_info.csv'
    )
    paragraphs = pd.read_csv(paragraph_path).set_index('paraID')

    tokens = tokens.sort_values('token_ID_within_document')
    token_ids = [int(token_id) for token_id in tokens['token_ID_within_document']]
    token_texts = [str(token) for token in tokens['word']]
    token_positions = {token_id: index for index, token_id in enumerate(token_ids)}
    token_paragraph_ids = [int(value) for value in tokens['paragraph_ID']]
    token_sentence_ids = [int(value) for value in tokens['sentence_ID']]
    token_byte_onsets = [int(value) for value in tokens['byte_onset']]
    token_byte_offsets = [int(value) for value in tokens['byte_offset']]

    entities_by_coref = {
        coref: group
        for coref, group in entities.groupby('COREF', sort=False)
    }
    cluster_rows = []
    status_counts = Counter()
    person_status_counts = Counter()
    category_counts = Counter()
    person_cluster_count = 0
    llm_candidate_count = 0

    for row in matches.itertuples(index=False):
        coref = int(row.COREF)
        candidate_ids = _parse_match_ids(row.matches)
        pdnc_id = int(row.pdncID)
        group = entities_by_coref.get(coref, entities.iloc[0:0])

        if pdnc_id != -1:
            status = 'resolved'
        elif len(candidate_ids) > 1:
            status = 'ambiguous'
        elif len(candidate_ids) == 1:
            status = 'unresolved_single_candidate'
        else:
            status = 'unmatched'

        mentions = []
        categories = set()
        for mention_row in group.sort_values('start_token').itertuples(index=False):
            category = str(mention_row.cat).upper()
            categories.add(category)
            start_token = int(mention_row.start_token)
            end_token = int(mention_row.end_token)
            start_pos = token_positions.get(start_token)
            context, context_available = _mention_context(
                start_token,
                end_token,
                token_texts,
                token_positions,
                context_window,
            )
            mentions.append({
                'text': str(mention_row.text),
                'prop': str(mention_row.prop),
                'category': category,
                'start_token': start_token,
                'end_token': end_token,
                'paragraph_id': (
                    token_paragraph_ids[start_pos] if start_pos is not None else None
                ),
                'sentence_id': (
                    token_sentence_ids[start_pos] if start_pos is not None else None
                ),
                'start_byte': (
                    token_byte_onsets[start_pos] if start_pos is not None else None
                ),
                'end_byte': (
                    token_byte_offsets[token_positions[end_token]]
                    if start_pos is not None and end_token in token_positions
                    else None
                ),
                'chapter_id': (
                    int(paragraphs.loc[token_paragraph_ids[start_pos] + 1, 'chapID'])
                    if start_pos is not None
                    and token_paragraph_ids[start_pos] + 1 in paragraphs.index
                    else None
                ),
                'context': context,
                'context_available': context_available,
            })

        contains_person = 'PER' in categories
        llm_candidate = status != 'resolved' and contains_person
        status_counts[status] += 1
        category_counts.update(categories)
        if contains_person:
            person_cluster_count += 1
            person_status_counts[status] += 1
        if llm_candidate:
            llm_candidate_count += 1

        candidate_names = []
        for candidate_id in candidate_ids:
            names = char_info['id2names'].get(candidate_id, [])
            if not isinstance(names, list):
                names = [names]
            candidate_names.append({
                'pdncID': candidate_id,
                'names': [str(name) for name in names],
            })

        cluster_rows.append({
            'novel': novel,
            'COREF': coref,
            'resolution_status': status,
            'pdncID': pdnc_id,
            'candidate_pdnc_ids': json.dumps(candidate_ids),
            'candidate_character_names': json.dumps(
                candidate_names, ensure_ascii=False
            ),
            'mention_count': len(mentions),
            'categories': json.dumps(sorted(categories)),
            'contains_person': contains_person,
            'llm_candidate': llm_candidate,
            'mentions': json.dumps(mentions, ensure_ascii=False),
        })

    clusters_df = pd.DataFrame(cluster_rows)
    clusters_path = os.path.join(novel_output, 'coref_clusters_for_review.csv')
    clusters_df.to_csv(clusters_path, index=False)
    character_catalog = []
    for character_id in sorted(int(key) for key in char_info['id2names']):
        names = char_info['id2names'][character_id]
        if not isinstance(names, list):
            names = [names]
        character_catalog.append({
            'pdncID': character_id,
            'canonical_name': str(char_info['id2parent'].get(character_id, '')),
            'names': [str(name) for name in names],
        })

    llm_context_packages = [
        {
            'COREF': row['COREF'],
            'resolution_status': row['resolution_status'],
            'existing_candidate_pdnc_ids': json.loads(row['candidate_pdnc_ids']),
            'existing_candidate_character_names': json.loads(
                row['candidate_character_names']
            ),
            'mention_count': row['mention_count'],
            'categories': json.loads(row['categories']),
            'mentions': json.loads(row['mentions']),
        }
        for row in cluster_rows
        if row['llm_candidate']
    ]
    context_package_path = os.path.join(
        novel_output, 'coref_llm_context_packages.json'
    )
    context_package = {
        'schema_version': 1,
        'novel': novel,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_policy': {
            'eligible_clusters': (
                'Unresolved clusters with at least one PER-category mention.'
            ),
            'allowed_character_ids': [
                character['pdncID'] for character in character_catalog
            ],
            'abstention_allowed': True,
            'existing_candidate_ids_are': (
                'Name-matching candidates from coref_matches.csv; an empty list '
                'does not mean the final resolver must abstain.'
            ),
        },
        'context_window_tokens_per_side': context_window,
        'source_files': {
            'entities': entities_path,
            'tokens': tokens_path,
            'coref_matches': matches_path,
            'paragraphs': paragraph_path,
        },
        'character_catalog': character_catalog,
        'clusters': llm_context_packages,
    }
    with open(context_package_path, 'w', encoding='utf-8') as package_file:
        json.dump(context_package, package_file, indent=2, ensure_ascii=False)

    cluster_count = len(cluster_rows)
    mention_counts = [row['mention_count'] for row in cluster_rows]
    unresolved_count = sum(
        count for status, count in status_counts.items() if status != 'resolved'
    )
    person_unresolved_count = sum(
        count for status, count in person_status_counts.items() if status != 'resolved'
    )
    summary = {
        'novel': novel,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_files': {
            'entities': entities_path,
            'tokens': tokens_path,
            'coref_matches': matches_path,
        },
        'context_window_tokens_per_side': context_window,
        'cluster_count': cluster_count,
        'mention_count': sum(mention_counts),
        'mean_mentions_per_cluster': (
            float(pd.Series(mention_counts).mean()) if mention_counts else 0
        ),
        'median_mentions_per_cluster': (
            float(pd.Series(mention_counts).median()) if mention_counts else 0
        ),
        'status_counts': dict(status_counts),
        'unresolved_cluster_count': unresolved_count,
        'unresolved_cluster_rate': unresolved_count / cluster_count if cluster_count else 0,
        'person_cluster_count': person_cluster_count,
        'person_cluster_status_counts': dict(person_status_counts),
        'unresolved_person_cluster_count': person_unresolved_count,
        'unresolved_person_cluster_rate': (
            person_unresolved_count / person_cluster_count if person_cluster_count else 0
        ),
        'llm_candidate_cluster_count': llm_candidate_count,
        'cluster_category_counts': dict(category_counts),
    }
    summary_path = os.path.join(novel_output, 'coref_resolution_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print("Wrote cluster report: {}".format(clusters_path))
    print("Wrote resolution summary: {}".format(summary_path))
    print("Wrote LLM context packages: {}".format(context_package_path))
    print("Clusters: {}; unresolved: {}; unresolved PER-containing: {}; LLM candidates: {}".format(
        cluster_count,
        unresolved_count,
        person_unresolved_count,
        llm_candidate_count,
    ))
    return summary


def _metric_distribution(values):
    metrics = pd.Series(values, dtype='float64')
    if metrics.empty:
        return {
            'n': 0,
            'mean': None,
            'median': None,
            'std_population': None,
            'min': None,
            'q1': None,
            'q3': None,
            'max': None,
        }

    return {
        'n': int(metrics.count()),
        'mean': float(metrics.mean()),
        'median': float(metrics.median()),
        'std_population': float(metrics.std(ddof=0)),
        'min': float(metrics.min()),
        'q1': float(metrics.quantile(0.25)),
        'q3': float(metrics.quantile(0.75)),
        'max': float(metrics.max()),
    }


def generate_population_report(novels=None, context_window=40, refresh_matches=False):
    """Generate individual cluster reports and aggregate statistics across novels."""
    novels = list(create_book_corefs.novels if novels is None else novels)
    if not novels:
        raise ValueError("At least one novel is required")
    unknown_novels = sorted(set(novels) - set(create_book_corefs.novels))
    if unknown_novels:
        raise ValueError("Unknown novels: {}".format(", ".join(unknown_novels)))

    summaries = [
        generate_coref_resolution_report(
            novel,
            context_window=context_window,
            refresh_matches=refresh_matches,
        )
        for novel in novels
    ]
    per_novel = pd.DataFrame(summaries)

    metric_columns = [
        'cluster_count',
        'mention_count',
        'mean_mentions_per_cluster',
        'median_mentions_per_cluster',
        'unresolved_cluster_count',
        'unresolved_cluster_rate',
        'person_cluster_count',
        'unresolved_person_cluster_count',
        'unresolved_person_cluster_rate',
        'llm_candidate_cluster_count',
    ]
    metric_statistics = {
        metric: _metric_distribution(per_novel[metric].tolist())
        for metric in metric_columns
    }
    totals = {
        metric: int(per_novel[metric].sum())
        for metric in (
            'cluster_count',
            'mention_count',
            'unresolved_cluster_count',
            'person_cluster_count',
            'unresolved_person_cluster_count',
            'llm_candidate_cluster_count',
        )
    }
    totals['unresolved_cluster_rate'] = (
        totals['unresolved_cluster_count'] / totals['cluster_count']
        if totals['cluster_count'] else 0
    )
    totals['unresolved_person_cluster_rate'] = (
        totals['unresolved_person_cluster_count'] / totals['person_cluster_count']
        if totals['person_cluster_count'] else 0
    )

    status_names = sorted({
        status
        for summary in summaries
        for status in summary['status_counts']
    })
    person_status_names = sorted({
        status
        for summary in summaries
        for status in summary['person_cluster_status_counts']
    })
    for status in status_names:
        totals['status_' + status] = int(sum(
            summary['status_counts'].get(status, 0) for summary in summaries
        ))
    for status in person_status_names:
        totals['person_status_' + status] = int(sum(
            summary['person_cluster_status_counts'].get(status, 0)
            for summary in summaries
        ))

    output_dir = create_book_corefs.COREF_ROOT
    per_novel_table = per_novel[
        ['novel'] + metric_columns
    ].copy()
    for status in status_names:
        per_novel_table['status_' + status] = per_novel['status_counts'].map(
            lambda counts: counts.get(status, 0)
        )
    for status in person_status_names:
        per_novel_table['person_status_' + status] = per_novel[
            'person_cluster_status_counts'
        ].map(lambda counts: counts.get(status, 0))
    per_novel_path = os.path.join(output_dir, 'coref_population_by_novel.csv')
    per_novel_table.to_csv(per_novel_path, index=False)
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'novel_count': len(summaries),
        'novels': novels,
        'context_window_tokens_per_side': context_window,
        'rate_aggregation': {
            'macro': 'Mean and median are calculated across per-novel rates.',
            'pooled': 'Pooled rates use summed unresolved counts divided by summed denominators.',
        },
        'standard_deviation': 'Population standard deviation (ddof=0) across novels.',
        'per_novel_statistics': metric_statistics,
        'pooled_totals_and_rates': totals,
        'per_novel_report': per_novel_path,
    }
    summary_path = os.path.join(output_dir, 'coref_population_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(report, summary_file, indent=2, ensure_ascii=False)

    print("Wrote per-novel population table: {}".format(per_novel_path))
    print("Wrote population summary: {}".format(summary_path))
    print("Novels: {}; clusters: {}; person clusters: {}; unresolved person clusters: {} ({:.1%} pooled)".format(
        len(summaries),
        totals['cluster_count'],
        totals['person_cluster_count'],
        totals['unresolved_person_cluster_count'],
        totals['unresolved_person_cluster_rate'],
    ))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate coreference cluster reports and population metrics."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--novel', choices=create_book_corefs.novels)
    selection.add_argument(
        '--all-novels',
        action='store_true',
        help='Generate reports and population statistics for every PDNC novel.',
    )
    parser.add_argument('--context-window', type=int, default=40)
    parser.add_argument(
        '--refresh-matches',
        action='store_true',
        help='Rebuild the existing name-to-PDNC match files before reporting.',
    )
    args = parser.parse_args()
    if args.all_novels:
        generate_population_report(
            context_window=args.context_window,
            refresh_matches=args.refresh_matches,
        )
    else:
        generate_coref_resolution_report(
            args.novel,
            context_window=args.context_window,
            refresh_matches=args.refresh_matches,
        )


if __name__ == '__main__':
    main()
