import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd

from booknlpen import create_book_corefs


QUOTE_MARKS = {'"', '“', '”', "'"}


def _as_int(value):
    if pd.isna(value):
        return None
    return int(value)


def _gold_speakers_by_byte(novel):
    quote_info = create_book_corefs.read_quote_df(novel)
    speakers_by_byte = defaultdict(set)

    for row in quote_info.itertuples(index=False):
        if pd.isna(row.speakerID):
            continue
        start_byte, end_byte = create_book_corefs.get_offset_bytes(
            str(row.qText), int(row.startByte), int(row.endByte)
        )
        speaker_id = int(row.speakerID)
        for byte_position in range(start_byte, end_byte):
            speakers_by_byte[byte_position].add(speaker_id)

    return speakers_by_byte


def _gold_speakers_for_quote(start_byte, end_byte, speakers_by_byte):
    speaker_ids = set()
    for byte_position in range(start_byte, end_byte):
        speaker_ids.update(speakers_by_byte.get(byte_position, ()))
    return sorted(speaker_ids)


def analyze_novel_assignments(novel):
    """Analyze the coreference assignments used as speakers in BookNLP quotes."""
    if novel not in create_book_corefs.novels:
        raise ValueError("Unknown novel: {}".format(novel))

    novel_output = os.path.join(create_book_corefs.COREF_ROOT, novel)
    quotes_path = os.path.join(novel_output, novel + '.quotes')
    entities_path = os.path.join(novel_output, novel + '.entities')
    tokens_path = os.path.join(novel_output, novel + '.tokens')
    matches_path = os.path.join(novel_output, 'coref_matches.csv')
    required_paths = [quotes_path, entities_path, tokens_path, matches_path]
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            "Missing BookNLP or matching input files: {}".format(
                ', '.join(missing_paths)
            )
        )

    quotes = create_book_corefs.read_booknlp_df(quotes_path)
    entities = create_book_corefs.read_booknlp_df(entities_path)
    tokens = create_book_corefs.read_booknlp_df(tokens_path).set_index(
        'token_ID_within_document'
    )
    matches = pd.read_csv(matches_path)
    char_info = create_book_corefs.read_char_info(novel)
    speakers_by_byte = _gold_speakers_by_byte(novel)

    coref_matches = {
        int(row.COREF): int(row.pdncID)
        for row in matches.itertuples(index=False)
    }
    entity_corefs_by_span = defaultdict(set)
    for row in entities.itertuples(index=False):
        entity_corefs_by_span[(int(row.start_token), int(row.end_token))].add(
            int(row.COREF)
        )

    quote_rows = []
    cluster_quote_counts = Counter()
    cluster_gold_correct = Counter()
    cluster_gold_comparable = Counter()
    attribution_status_counts = Counter()

    for quote_index, row in enumerate(quotes.itertuples(index=False)):
        quote_start = int(row.quote_start)
        quote_end = int(row.quote_end)
        start_token = tokens.loc[quote_start]
        end_token = tokens.loc[quote_end]

        if str(start_token['word']) in QUOTE_MARKS:
            quote_content_start = min(quote_start + 1, quote_end)
        else:
            quote_content_start = quote_start
        if str(end_token['word']) in QUOTE_MARKS:
            quote_content_end = max(quote_end - 1, quote_content_start)
        else:
            quote_content_end = quote_end

        content_start = tokens.loc[quote_content_start]
        content_end = tokens.loc[quote_content_end]
        start_byte = int(content_start['byte_onset'])
        end_byte = int(content_end['byte_offset'])
        gold_speaker_ids = _gold_speakers_for_quote(
            start_byte, end_byte, speakers_by_byte
        )

        mention_start = _as_int(row.mention_start)
        mention_end = _as_int(row.mention_end)
        cluster_id = _as_int(row.char_id)
        if cluster_id is None or mention_start is None or mention_end is None:
            cluster_id = None
            pdnc_id = None
            character_name = None
            status = 'no_attributed_mention'
            mention_matches_assignment = None
        else:
            pdnc_id = coref_matches.get(cluster_id)
            if pdnc_id is None:
                status = 'missing_cluster_match'
                character_name = None
            elif pdnc_id < 0:
                status = 'unresolved_cluster'
                character_name = None
            else:
                status = 'resolved_cluster'
                character_name = str(
                    char_info['id2parent'].get(
                        pdnc_id,
                        char_info['id2names'].get(pdnc_id, ''),
                    )
                )
            mention_matches_assignment = (
                cluster_id in entity_corefs_by_span.get(
                    (mention_start, mention_end), set()
                )
            )
            cluster_quote_counts[cluster_id] += 1

        is_gold_comparable = pdnc_id is not None and pdnc_id >= 0 and bool(
            gold_speaker_ids
        )
        is_gold_correct = (
            pdnc_id in gold_speaker_ids if is_gold_comparable else None
        )
        if is_gold_comparable:
            cluster_gold_comparable[cluster_id] += 1
            if is_gold_correct:
                cluster_gold_correct[cluster_id] += 1

        attribution_status_counts[status] += 1
        quote_rows.append({
            'novel': novel,
            'quote_index': quote_index,
            'quote_start_token': quote_start,
            'quote_end_token': quote_end,
            'quote_start_byte': start_byte,
            'quote_end_byte': end_byte,
            'quote_text': str(row.quote),
            'selected_mention': (
                str(row.mention_phrase) if mention_start is not None else None
            ),
            'mention_start_token': mention_start,
            'mention_end_token': mention_end,
            'speaker_cluster_id': cluster_id,
            'speaker_resolution_status': status,
            'predicted_pdnc_id': pdnc_id if pdnc_id is not None and pdnc_id >= 0 else None,
            'predicted_character_name': character_name,
            'gold_speaker_ids': json.dumps(gold_speaker_ids),
            'gold_comparable': is_gold_comparable,
            'gold_correct_if_resolved': is_gold_correct,
            'selected_mention_matches_assignment': mention_matches_assignment,
        })

    quote_df = pd.DataFrame(quote_rows)
    quote_output_path = os.path.join(
        novel_output, 'speaker_assignments_by_quote.csv'
    )
    quote_df.to_csv(quote_output_path, index=False)

    cluster_rows = []
    for cluster_id, quote_count in cluster_quote_counts.items():
        pdnc_id = coref_matches.get(cluster_id)
        if pdnc_id is None:
            status = 'missing_cluster_match'
            character_name = None
        elif pdnc_id < 0:
            status = 'unresolved_cluster'
            character_name = None
        else:
            status = 'resolved_cluster'
            character_name = str(
                char_info['id2parent'].get(
                    pdnc_id,
                    char_info['id2names'].get(pdnc_id, ''),
                )
            )
        comparable_count = cluster_gold_comparable[cluster_id]
        correct_count = cluster_gold_correct[cluster_id]
        cluster_rows.append({
            'novel': novel,
            'speaker_cluster_id': cluster_id,
            'speaker_resolution_status': status,
            'pdnc_id': pdnc_id if pdnc_id is not None and pdnc_id >= 0 else None,
            'character_name': character_name,
            'attributed_quote_count': quote_count,
            'gold_comparable_quote_count': comparable_count,
            'gold_correct_quote_count': correct_count,
            'gold_accuracy_when_resolved': (
                correct_count / comparable_count if comparable_count else None
            ),
        })

    cluster_columns = [
        'novel',
        'speaker_cluster_id',
        'speaker_resolution_status',
        'pdnc_id',
        'character_name',
        'attributed_quote_count',
        'gold_comparable_quote_count',
        'gold_correct_quote_count',
        'gold_accuracy_when_resolved',
    ]
    cluster_df = pd.DataFrame(cluster_rows, columns=cluster_columns).sort_values(
        'attributed_quote_count', ascending=False
    )
    cluster_output_path = os.path.join(
        novel_output, 'speaker_clusters_used_as_speakers.csv'
    )
    cluster_df.to_csv(cluster_output_path, index=False)

    quote_count = len(quote_df)
    attributed_count = int(
        quote_df['speaker_cluster_id'].notna().sum()
    )
    resolved_count = int(
        quote_df['speaker_resolution_status'].eq('resolved_cluster').sum()
    )
    unresolved_count = int(
        quote_df['speaker_resolution_status'].eq('unresolved_cluster').sum()
    )
    missing_match_count = int(
        quote_df['speaker_resolution_status'].eq('missing_cluster_match').sum()
    )
    no_attribution_count = int(
        quote_df['speaker_resolution_status'].eq('no_attributed_mention').sum()
    )
    gold_comparable_count = int(quote_df['gold_comparable'].sum())
    gold_correct_count = int(
        quote_df['gold_correct_if_resolved'].eq(True).sum()
    )
    inconsistent_mention_count = int(
        quote_df['selected_mention_matches_assignment'].eq(False).sum()
    )
    summary = {
        'novel': novel,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'quote_count': quote_count,
        'attributed_quote_count': attributed_count,
        'quote_attribution_coverage': (
            attributed_count / quote_count if quote_count else 0
        ),
        'resolved_cluster_attribution_count': resolved_count,
        'unresolved_cluster_attribution_count': unresolved_count,
        'missing_cluster_match_attribution_count': missing_match_count,
        'no_attributed_mention_count': no_attribution_count,
        'unresolved_attribution_rate_all_quotes': (
            unresolved_count / quote_count if quote_count else 0
        ),
        'unresolved_attribution_rate_assigned_quotes': (
            unresolved_count / attributed_count if attributed_count else 0
        ),
        'resolved_attribution_rate_assigned_quotes': (
            resolved_count / attributed_count if attributed_count else 0
        ),
        'distinct_speaker_clusters': len(cluster_quote_counts),
        'distinct_unresolved_speaker_clusters': sum(
            coref_matches.get(cluster_id, -1) < 0
            for cluster_id in cluster_quote_counts
        ),
        'gold_comparable_resolved_attributions': gold_comparable_count,
        'gold_correct_resolved_attributions': gold_correct_count,
        'gold_accuracy_when_cluster_resolved': (
            gold_correct_count / gold_comparable_count
            if gold_comparable_count else None
        ),
        'selected_mention_assignment_mismatch_count': inconsistent_mention_count,
        'attribution_status_counts': dict(attribution_status_counts),
        'quote_level_report': quote_output_path,
        'cluster_level_report': cluster_output_path,
    }
    summary_path = os.path.join(
        novel_output, 'speaker_assignment_summary.json'
    )
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print(
        "{}: {} quotes; unresolved-cluster attributions: {}/{} ({:.1%}); "
        "gold accuracy among resolved: {}".format(
            novel,
            quote_count,
            unresolved_count,
            attributed_count,
            summary['unresolved_attribution_rate_assigned_quotes'],
            (
                "{:.1%}".format(summary['gold_accuracy_when_cluster_resolved'])
                if summary['gold_accuracy_when_cluster_resolved'] is not None
                else 'n/a'
            ),
        )
    )
    return summary


def generate_population_report(novels=None):
    novels = list(create_book_corefs.novels if novels is None else novels)
    if not novels:
        raise ValueError("At least one novel is required")
    unknown_novels = sorted(set(novels) - set(create_book_corefs.novels))
    if unknown_novels:
        raise ValueError("Unknown novels: {}".format(", ".join(unknown_novels)))

    summaries = [analyze_novel_assignments(novel) for novel in novels]
    per_novel = pd.DataFrame(summaries)
    per_novel_path = os.path.join(
        create_book_corefs.COREF_ROOT, 'speaker_assignment_population_by_novel.csv'
    )
    per_novel.to_csv(per_novel_path, index=False)

    total_quotes = sum(summary['quote_count'] for summary in summaries)
    total_attributed = sum(
        summary['attributed_quote_count'] for summary in summaries
    )
    total_resolved = sum(
        summary['resolved_cluster_attribution_count'] for summary in summaries
    )
    total_unresolved = sum(
        summary['unresolved_cluster_attribution_count'] for summary in summaries
    )
    total_missing = sum(
        summary['missing_cluster_match_attribution_count'] for summary in summaries
    )
    total_no_attribution = sum(
        summary['no_attributed_mention_count'] for summary in summaries
    )
    total_gold_comparable = sum(
        summary['gold_comparable_resolved_attributions'] for summary in summaries
    )
    total_gold_correct = sum(
        summary['gold_correct_resolved_attributions'] for summary in summaries
    )

    rate_columns = [
        'quote_attribution_coverage',
        'unresolved_attribution_rate_all_quotes',
        'unresolved_attribution_rate_assigned_quotes',
        'resolved_attribution_rate_assigned_quotes',
        'gold_accuracy_when_cluster_resolved',
    ]
    macro_statistics = {}
    for column in rate_columns:
        values = per_novel[column].dropna()
        macro_statistics[column] = {
            'n': int(values.count()),
            'mean': float(values.mean()) if not values.empty else None,
            'median': float(values.median()) if not values.empty else None,
            'std_population': (
                float(values.std(ddof=0)) if not values.empty else None
            ),
            'min': float(values.min()) if not values.empty else None,
            'max': float(values.max()) if not values.empty else None,
        }

    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'novel_count': len(summaries),
        'novels': novels,
        'rate_definitions': {
            'unresolved_attribution_rate_all_quotes': (
                'Quotes whose selected speaker cluster is unresolved / all BookNLP quotes.'
            ),
            'unresolved_attribution_rate_assigned_quotes': (
                'Quotes whose selected speaker cluster is unresolved / quotes with an attributed mention.'
            ),
            'gold_accuracy_when_cluster_resolved': (
                'Resolved cluster predictions in the PDNC gold speaker set / '
                'resolved predictions overlapping at least one gold quote.'
            ),
            'resolution_source': (
                'coref_matches.csv name matching; this is a post-hoc proxy, '
                'not BookNLP native character IDs.'
            ),
        },
        'pooled_totals_and_rates': {
            'quote_count': total_quotes,
            'attributed_quote_count': total_attributed,
            'quote_attribution_coverage': (
                total_attributed / total_quotes if total_quotes else 0
            ),
            'resolved_cluster_attribution_count': total_resolved,
            'unresolved_cluster_attribution_count': total_unresolved,
            'missing_cluster_match_attribution_count': total_missing,
            'no_attributed_mention_count': total_no_attribution,
            'unresolved_attribution_rate_all_quotes': (
                total_unresolved / total_quotes if total_quotes else 0
            ),
            'unresolved_attribution_rate_assigned_quotes': (
                total_unresolved / total_attributed if total_attributed else 0
            ),
            'resolved_attribution_rate_assigned_quotes': (
                total_resolved / total_attributed if total_attributed else 0
            ),
            'gold_comparable_resolved_attributions': total_gold_comparable,
            'gold_correct_resolved_attributions': total_gold_correct,
            'gold_accuracy_when_cluster_resolved': (
                total_gold_correct / total_gold_comparable
                if total_gold_comparable else None
            ),
        },
        'per_novel_macro_statistics': macro_statistics,
        'per_novel_report': per_novel_path,
    }
    summary_path = os.path.join(
        create_book_corefs.COREF_ROOT, 'speaker_assignment_population_summary.json'
    )
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(report, summary_file, indent=2, ensure_ascii=False)

    print("Wrote per-novel assignment table: {}".format(per_novel_path))
    print("Wrote assignment population summary: {}".format(summary_path))
    pooled = report['pooled_totals_and_rates']
    print(
        "Pooled: {} quotes; unresolved cluster attributions {}/{} ({:.1%}); "
        "resolved-cluster accuracy {}".format(
            total_quotes,
            total_unresolved,
            total_attributed,
            pooled['unresolved_attribution_rate_assigned_quotes'],
            (
                "{:.1%}".format(pooled['gold_accuracy_when_cluster_resolved'])
                if pooled['gold_accuracy_when_cluster_resolved'] is not None
                else 'n/a'
            ),
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BookNLP speaker clusters selected for each quote."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--novel', choices=create_book_corefs.novels)
    selection.add_argument(
        '--all-novels',
        action='store_true',
        help='Analyze speaker assignments for all PDNC novels.',
    )
    args = parser.parse_args()
    if args.all_novels:
        generate_population_report()
    else:
        analyze_novel_assignments(args.novel)


if __name__ == '__main__':
    main()
