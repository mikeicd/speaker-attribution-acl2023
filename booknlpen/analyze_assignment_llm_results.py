import argparse
import hashlib
import json
import os
import re
from collections import Counter

import pandas as pd

from booknlpen import create_book_corefs
from booknlpen.assignment_llm_artifacts import (
    context_package_filename,
    decision_limit_suffix,
    ollama_results_filename,
)


def _read_json(path):
    with open(path, encoding='utf-8') as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object in {}'.format(path))
    return value


def _parse_gold_speaker_ids(value, row_number):
    if isinstance(value, list):
        speaker_ids = value
    elif isinstance(value, str):
        try:
            speaker_ids = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                'Invalid gold_speaker_ids on quote report row {}'.format(
                    row_number
                )
            ) from exc
    else:
        speaker_ids = []
    if not isinstance(speaker_ids, list):
        raise ValueError(
            'gold_speaker_ids must be a JSON list on quote report row {}'.format(
                row_number
            )
        )
    try:
        return sorted({int(speaker_id) for speaker_id in speaker_ids})
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'Gold speaker IDs must be integers on quote report row {}'.format(
                row_number
            )
        ) from exc


def _optional_int(value, field, row_number):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        raise ValueError(
            '{} must be an integer on quote report row {}'.format(
                field, row_number
            )
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            '{} must be an integer on quote report row {}'.format(
                field, row_number
            )
        ) from exc


def _safe_model_label(model):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', model)


def _load_expected_assignments(package, novel):
    if package.get('schema_version') != 2 or package.get('novel') != novel:
        raise ValueError('Unsupported or mismatched context package')
    passages = package.get('passages')
    if not isinstance(passages, list):
        raise ValueError('Context package passages must be a list')

    pending = {}
    passage_expected = {}
    for passage in passages:
        passage_id = passage.get('passage_id')
        assignments = passage.get('assignments')
        if not isinstance(passage_id, str) or not isinstance(assignments, list):
            raise ValueError('Invalid passage entry in context package')
        decision_assignments = [
            assignment for assignment in assignments
            if assignment.get('decision_required')
        ]
        expected_count = passage.get('expected_decision_count')
        if expected_count != len(decision_assignments):
            raise ValueError(
                'Passage {} declares {} decisions but contains {}'.format(
                    passage_id, expected_count, len(decision_assignments)
                )
            )
        passage_expected[passage_id] = set()
        for assignment in decision_assignments:
            assignment_id = assignment.get('assignment_id')
            quote_index = assignment.get('quote_index')
            if (
                not isinstance(assignment_id, str)
                or not isinstance(quote_index, int)
                or isinstance(quote_index, bool)
            ):
                raise ValueError(
                    'Invalid pending assignment in passage {}'.format(passage_id)
                )
            expected_id = '{}:quote:{}'.format(novel, quote_index)
            if assignment_id != expected_id:
                raise ValueError(
                    'Assignment ID {} does not match quote index {}'.format(
                        assignment_id, quote_index
                    )
                )
            if assignment_id in pending:
                raise ValueError(
                    'Duplicate pending assignment in package: {}'.format(
                        assignment_id
                    )
                )
            pending[assignment_id] = {
                'assignment_id': assignment_id,
                'quote_index': quote_index,
                'cluster_key': assignment.get('cluster_key'),
                'COREF': assignment.get('COREF'),
                'resolution_status': assignment.get('resolution_status'),
                'passage_id': passage_id,
            }
            passage_expected[passage_id].add(assignment_id)

    expected_count = package.get('expected_output_decision_count')
    if expected_count != len(pending):
        raise ValueError(
            'Context package declares {} decisions but contains {}'.format(
                expected_count, len(pending)
            )
        )
    return pending, passage_expected


def _load_llm_decisions(results, novel, pending, passage_expected, character_ids):
    if results.get('schema_version') != 2:
        raise ValueError('Unsupported Ollama results schema')
    if results.get('decision_policy') != 'forced_choice':
        raise ValueError(
            'Expected forced_choice results; abstention-policy results cannot '
            'be compared by this script'
        )
    if not isinstance(results.get('passages'), dict):
        raise ValueError('Ollama results passages must be an object')
    if results.get('novel') != novel:
        raise ValueError('Ollama results novel does not match context package')

    decisions_by_id = {}
    for passage_id, passage_result in results['passages'].items():
        if passage_id not in passage_expected:
            raise ValueError(
                'Results contain a passage absent from the package: {}'.format(
                    passage_id
                )
            )
        if not isinstance(passage_result, dict):
            raise ValueError(
                'Passage {} result must be an object'.format(passage_id)
            )
        decisions = passage_result.get('decisions')
        if not isinstance(decisions, list):
            raise ValueError(
                'Passage {} decisions must be a list'.format(passage_id)
            )
        expected_ids = passage_expected[passage_id]
        if passage_result.get('expected_decision_count') != len(expected_ids):
            raise ValueError(
                'Result expected-decision count does not match passage {}'.format(
                    passage_id
                )
            )
        if len(decisions) != len(expected_ids):
            raise ValueError(
                'Passage {} has {} decisions; expected {}'.format(
                    passage_id, len(decisions), len(expected_ids)
                )
            )
        seen_in_passage = set()
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError(
                    'Each result decision must be an object in {}'.format(
                        passage_id
                    )
                )
            assignment_id = decision.get('assignment_id')
            if assignment_id not in expected_ids:
                raise ValueError(
                    'Unexpected decision {} in passage {}'.format(
                        assignment_id, passage_id
                    )
                )
            if assignment_id in seen_in_passage or assignment_id in decisions_by_id:
                raise ValueError(
                    'Duplicate result decision: {}'.format(assignment_id)
                )
            if decision.get('decision') != 'resolve':
                raise ValueError(
                    'Forced-choice decision is not resolve: {}'.format(
                        assignment_id
                    )
                )
            pdnc_id = decision.get('pdncID')
            if (
                not isinstance(pdnc_id, int)
                or isinstance(pdnc_id, bool)
                or pdnc_id not in character_ids
            ):
                raise ValueError(
                    'Decision {} has an invalid pdncID'.format(assignment_id)
                )
            if decision.get('cluster_key') != pending[assignment_id]['cluster_key']:
                raise ValueError(
                    'Decision {} has a mismatched cluster_key'.format(
                        assignment_id
                    )
                )
            if decision.get('COREF') != pending[assignment_id]['COREF']:
                raise ValueError(
                    'Decision {} has a mismatched COREF'.format(assignment_id)
                )
            confidence = decision.get('confidence')
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0 <= confidence <= 1
            ):
                raise ValueError(
                    'Decision {} has an invalid confidence'.format(assignment_id)
                )
            if not isinstance(decision.get('rationale'), str):
                raise ValueError(
                    'Decision {} has an invalid rationale'.format(assignment_id)
                )
            decisions_by_id[assignment_id] = decision
            seen_in_passage.add(assignment_id)

    return decisions_by_id


def _load_quote_rows(quote_report_path, novel, pending, decisions_by_id):
    if not os.path.isfile(quote_report_path):
        raise FileNotFoundError(
            'Baseline quote report not found: {}. Run '
            'booknlpen.analyze_speaker_assignments first.'.format(
                quote_report_path
            )
        )
    quote_df = pd.read_csv(quote_report_path)
    required_columns = {
        'novel',
        'quote_index',
        'quote_text',
        'speaker_cluster_id',
        'speaker_resolution_status',
        'cluster_resolution_detail',
        'predicted_pdnc_id',
        'gold_speaker_ids',
    }
    missing_columns = required_columns - set(quote_df.columns)
    if missing_columns:
        raise ValueError(
            'Baseline quote report is missing columns: {}'.format(
                ', '.join(sorted(missing_columns))
            )
        )
    if set(quote_df['novel'].dropna()) != {novel}:
        raise ValueError('Baseline quote report novel does not match {}'.format(novel))
    if quote_df['quote_index'].duplicated().any():
        raise ValueError('Baseline quote report contains duplicate quote_index')

    rows_by_id = {}
    for row_number, row in enumerate(quote_df.to_dict(orient='records'), start=2):
        quote_index = _optional_int(row['quote_index'], 'quote_index', row_number)
        assignment_id = '{}:quote:{}'.format(novel, quote_index)
        gold_ids = _parse_gold_speaker_ids(row['gold_speaker_ids'], row_number)
        baseline_pdnc_id = _optional_int(
            row['predicted_pdnc_id'], 'predicted_pdnc_id', row_number
        )
        llm_decision = decisions_by_id.get(assignment_id)
        target = pending.get(assignment_id)
        llm_pdnc_id = llm_decision['pdncID'] if llm_decision else None
        if target is not None and baseline_pdnc_id is not None:
            raise ValueError(
                'LLM target {} already has a baseline resolved prediction'.format(
                    assignment_id
                )
            )
        combined_pdnc_id = (
            llm_pdnc_id if target is not None else baseline_pdnc_id
        )
        rows_by_id[assignment_id] = {
            'novel': novel,
            'assignment_id': assignment_id,
            'quote_index': quote_index,
            'quote_text': row['quote_text'],
            'speaker_cluster_id': _optional_int(
                row['speaker_cluster_id'], 'speaker_cluster_id', row_number
            ),
            'speaker_resolution_status': row['speaker_resolution_status'],
            'cluster_resolution_detail': row['cluster_resolution_detail'],
            'is_llm_target': target is not None,
            'llm_result_status': (
                'completed' if llm_decision
                else 'pending' if target is not None
                else 'not_target'
            ),
            'llm_pdnc_id': llm_pdnc_id,
            'llm_confidence': (
                llm_decision.get('confidence') if llm_decision else None
            ),
            'llm_rationale': (
                llm_decision.get('rationale') if llm_decision else None
            ),
            'baseline_pdnc_id': baseline_pdnc_id,
            'combined_pdnc_id': combined_pdnc_id,
            'gold_speaker_ids': json.dumps(gold_ids),
            'gold_comparable': bool(gold_ids),
            'baseline_correct': (
                baseline_pdnc_id in gold_ids
                if baseline_pdnc_id is not None and gold_ids else None
            ),
            'llm_correct': (
                llm_pdnc_id in gold_ids
                if llm_pdnc_id is not None and gold_ids else None
            ),
            'combined_correct': (
                combined_pdnc_id in gold_ids
                if combined_pdnc_id is not None and gold_ids else None
            ),
        }

    expected_quote_ids = {
        '{}:quote:{}'.format(novel, index)
        for index in range(len(rows_by_id))
    }
    if set(rows_by_id) != expected_quote_ids:
        raise ValueError(
            'Baseline quote indices must be contiguous and start at zero'
        )
    unknown_targets = set(pending) - set(rows_by_id)
    unknown_decisions = set(decisions_by_id) - set(rows_by_id)
    if unknown_targets or unknown_decisions:
        raise ValueError(
            'Package/results include assignments absent from baseline quote report'
        )
    return [rows_by_id[key] for key in sorted(
        rows_by_id, key=lambda assignment_id: rows_by_id[assignment_id]['quote_index']
    )]


def _score(rows, prediction_field):
    comparable_rows = [row for row in rows if row['gold_comparable']]
    predicted_rows = [
        row for row in comparable_rows if row[prediction_field] is not None
    ]
    correct_count = sum(
        row[prediction_field] in json.loads(row['gold_speaker_ids'])
        for row in predicted_rows
    )
    comparable_count = len(comparable_rows)
    predicted_count = len(predicted_rows)
    return {
        'gold_comparable_quote_count': comparable_count,
        'predicted_gold_comparable_quote_count': predicted_count,
        'correct_prediction_count': correct_count,
        'accuracy_when_predicted': (
            correct_count / predicted_count if predicted_count else None
        ),
        'prediction_coverage_over_gold_comparable': (
            predicted_count / comparable_count if comparable_count else 0
        ),
        'correct_rate_over_all_gold_comparable': (
            correct_count / comparable_count if comparable_count else 0
        ),
    }


def _usage_summary(results, completed_decision_count):
    usage_totals = Counter()
    valid_usage = Counter()
    invalid_usage = Counter()
    attempt_count = 0
    invalid_attempt_count = 0
    for attempts in results.get('attempts_by_passage', {}).values():
        if not isinstance(attempts, list):
            raise ValueError('attempts_by_passage values must be lists')
        for attempt in attempts:
            attempt_count += 1
            is_valid = attempt.get('status') == 'valid'
            if not is_valid:
                invalid_attempt_count += 1
            usage = attempt.get('usage', {})
            if not isinstance(usage, dict):
                raise ValueError('Attempt usage must be an object')
            for field in (
                'prompt_eval_count',
                'eval_count',
                'total_duration',
                'prompt_eval_duration',
                'eval_duration',
                'load_duration',
            ):
                value = usage.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage_totals[field] += value
                    (valid_usage if is_valid else invalid_usage)[field] += value

    total_tokens = (
        usage_totals['prompt_eval_count'] + usage_totals['eval_count']
    )
    valid_tokens = valid_usage['prompt_eval_count'] + valid_usage['eval_count']
    invalid_tokens = (
        invalid_usage['prompt_eval_count'] + invalid_usage['eval_count']
    )
    return {
        'attempt_count': attempt_count,
        'invalid_attempt_count': invalid_attempt_count,
        'prompt_tokens_all_attempts': usage_totals['prompt_eval_count'],
        'generated_tokens_all_attempts': usage_totals['eval_count'],
        'total_tokens_all_attempts': total_tokens,
        'prompt_tokens_valid_attempts': valid_usage['prompt_eval_count'],
        'generated_tokens_valid_attempts': valid_usage['eval_count'],
        'total_tokens_valid_attempts': valid_tokens,
        'total_tokens_invalid_attempts': invalid_tokens,
        'api_duration_seconds_all_attempts': (
            usage_totals['total_duration'] / 1_000_000_000
        ),
        'tokens_per_completed_decision_all_attempts': (
            total_tokens / completed_decision_count
            if completed_decision_count else None
        ),
    }


def analyze_novel_results(
    novel,
    model='qwen2.5:7b',
    results_path=None,
    quote_report_path=None,
    output_dir=None,
    max_decisions_per_passage=None,
):
    if novel not in create_book_corefs.novels:
        raise ValueError('Unknown novel: {}'.format(novel))
    novel_output = os.path.join(create_book_corefs.COREF_ROOT, novel)
    model_label = _safe_model_label(model)
    package_path = os.path.join(
        novel_output, context_package_filename(max_decisions_per_passage)
    )
    results_path = results_path or os.path.join(
        novel_output,
        ollama_results_filename(
            model_label, max_decisions_per_passage
        ),
    )
    quote_report_path = quote_report_path or os.path.join(
        novel_output, 'speaker_assignments_by_quote.csv'
    )
    output_dir = output_dir or novel_output

    for path in (package_path, results_path):
        if not os.path.isfile(path):
            raise FileNotFoundError('Required input file not found: {}'.format(path))
    with open(package_path, 'rb') as package_file:
        package_bytes = package_file.read()
    package = json.loads(package_bytes.decode('utf-8'))
    if package.get('max_decisions_per_passage') != max_decisions_per_passage:
        raise ValueError(
            'Context package decision limit does not match the requested limit'
        )
    pending, passage_expected = _load_expected_assignments(package, novel)
    results = _read_json(results_path)
    if results.get('source_package_sha256') != hashlib.sha256(
        package_bytes
    ).hexdigest():
        raise ValueError(
            'Ollama results were generated from a different context package'
        )
    if results.get('model') != model:
        raise ValueError('Ollama results model does not match {}'.format(model))
    if results.get('novel') != novel:
        raise ValueError('Ollama results novel does not match {}'.format(novel))
    if results.get('expected_decision_count') != len(pending):
        raise ValueError(
            'Ollama expected-decision count does not match context package'
        )

    character_catalog = package.get('character_catalog')
    if not isinstance(character_catalog, list):
        raise ValueError('Context package character_catalog must be a list')
    try:
        character_ids = {
            int(character['pdncID']) for character in character_catalog
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Context package contains an invalid character catalog') from exc
    decisions_by_id = _load_llm_decisions(
        results, novel, pending, passage_expected, character_ids
    )
    rows = _load_quote_rows(
        quote_report_path, novel, pending, decisions_by_id
    )

    target_rows = [row for row in rows if row['is_llm_target']]
    successful_targets = [
        row for row in target_rows if row['llm_result_status'] == 'completed'
    ]
    baseline_metrics = _score(rows, 'baseline_pdnc_id')
    llm_metrics = _score(target_rows, 'llm_pdnc_id')
    combined_metrics = _score(rows, 'combined_pdnc_id')
    usage = _usage_summary(results, len(decisions_by_id))
    complete_passage_count = len(results['passages'])
    expected_passage_count = len(passage_expected)
    summary = {
        'schema_version': 1,
        'novel': novel,
        'model': model,
        'decision_policy': results['decision_policy'],
        'max_decisions_per_passage': max_decisions_per_passage,
        'generated_at_utc': results.get('generated_at_utc'),
        'analysis_generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'metric_definition': (
            'A quote is gold-comparable when its gold_speaker_ids list is '
            'non-empty. A prediction is correct when its pdncID is in that '
            'list. accuracy_when_predicted excludes quotes without a prediction; '
            'correct_rate_over_all_gold_comparable counts uncovered quotes as '
            'incorrect.'
        ),
        'is_complete': (
            len(decisions_by_id) == len(pending)
            and complete_passage_count == expected_passage_count
            and not results.get('failed_passages')
        ),
        'input_files': {
            'context_package': os.path.relpath(package_path),
            'ollama_results': os.path.relpath(results_path),
            'baseline_quote_report': os.path.relpath(quote_report_path),
        },
        'counts': {
            'total_quotes': len(rows),
            'expected_llm_target_assignments': len(pending),
            'completed_llm_target_assignments': len(decisions_by_id),
            'pending_llm_target_assignments': len(pending) - len(decisions_by_id),
            'expected_passages': expected_passage_count,
            'completed_passages': complete_passage_count,
            'failed_passages': len(results.get('failed_passages', {})),
            'llm_target_assignments_with_gold': sum(
                row['gold_comparable'] for row in target_rows
            ),
            'llm_target_assignments_without_gold': sum(
                not row['gold_comparable'] for row in target_rows
            ),
        },
        'metrics': {
            'baseline_booknlp_resolved_assignments': baseline_metrics,
            'llm_on_target_assignments': llm_metrics,
            'combined_booknlp_plus_llm': combined_metrics,
            'combined_accuracy_delta_vs_baseline': (
                combined_metrics['correct_rate_over_all_gold_comparable']
                - baseline_metrics['correct_rate_over_all_gold_comparable']
            ),
        },
        'ollama_usage': usage,
        'output_files': {
            'summary_json': os.path.join(
                output_dir,
                'speaker_assignment_llm_comparison_{}{}.json'.format(
                    model_label,
                    decision_limit_suffix(max_decisions_per_passage),
                ),
            ),
            'quote_level_csv': os.path.join(
                output_dir,
                'speaker_assignment_llm_comparison_{}{}_by_quote.csv'.format(
                    model_label,
                    decision_limit_suffix(max_decisions_per_passage),
                ),
            ),
        },
    }
    summary_path = os.path.join(
        output_dir,
        'speaker_assignment_llm_comparison_{}{}.json'.format(
            model_label,
            decision_limit_suffix(max_decisions_per_passage),
        ),
    )
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        summary['output_files']['quote_level_csv'], index=False
    )
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        summary_file.write('\n')

    pending_count = len(pending) - len(decisions_by_id)
    print(
        '{} / {}: {}/{} LLM target assignments; {} pending; '
        'target accuracy {:.1%}; combined end-to-end accuracy {:.1%}; '
        'tokens {}'.format(
            novel,
            model,
            len(successful_targets),
            len(pending),
            pending_count,
            llm_metrics['correct_rate_over_all_gold_comparable'],
            combined_metrics['correct_rate_over_all_gold_comparable'],
            usage['total_tokens_all_attempts'],
        )
    )
    print('Wrote comparison summary: {}'.format(summary_path))
    print(
        'Wrote quote-level comparison: {}'.format(
            summary['output_files']['quote_level_csv']
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compare BookNLP speaker assignments with forced-choice Ollama '
            'predictions and evaluate both against PDNC speaker annotations.'
        )
    )
    parser.add_argument('--novel', required=True, choices=create_book_corefs.novels)
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument(
        '--max-decisions-per-passage',
        type=int,
        help='Analyze the matching capped context package and result files.',
    )
    parser.add_argument('--results', help='Path to an Ollama result JSON file.')
    parser.add_argument(
        '--quote-report',
        help='Path to speaker_assignments_by_quote.csv from the baseline analysis.',
    )
    parser.add_argument('--output-dir', help='Output directory for comparison reports.')
    args = parser.parse_args()
    analyze_novel_results(
        args.novel,
        model=args.model,
        results_path=args.results,
        quote_report_path=args.quote_report,
        output_dir=args.output_dir,
        max_decisions_per_passage=args.max_decisions_per_passage,
    )


if __name__ == '__main__':
    main()
