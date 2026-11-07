import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import median

from booknlpen import create_book_corefs
from booknlpen.assignment_llm_artifacts import (
    context_package_filename,
    decision_limit_suffix,
    ollama_results_filename,
)


SIZE_BUCKETS = (
    ('1-4', 1, 4),
    ('5-8', 5, 8),
    ('9-14', 9, 14),
    ('15+', 15, None),
)


def _read_json(path):
    with open(path, encoding='utf-8') as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object in {}'.format(path))
    return value


def _read_gold_speakers(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8', newline='') as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or not {
            'quote_index', 'gold_speaker_ids'
        }.issubset(reader.fieldnames):
            raise ValueError(
                'Quote report must contain quote_index and gold_speaker_ids: '
                '{}'.format(path)
            )
        gold_by_quote = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                quote_index = int(row['quote_index'])
                speaker_ids = json.loads(row['gold_speaker_ids'])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    'Invalid gold quote data on line {}'.format(line_number)
                ) from exc
            if not isinstance(speaker_ids, list):
                raise ValueError(
                    'gold_speaker_ids must be a list on line {}'.format(
                        line_number
                    )
                )
            if quote_index in gold_by_quote:
                raise ValueError('Duplicate quote_index in {}'.format(path))
            try:
                gold_by_quote[quote_index] = {
                    int(speaker_id) for speaker_id in speaker_ids
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'Gold speaker IDs must be integers on line {}'.format(
                        line_number
                    )
                ) from exc
    return gold_by_quote


def _safe_model_label(model):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', model)


def _bucket_for_size(size):
    for label, minimum, maximum in SIZE_BUCKETS:
        if size >= minimum and (maximum is None or size <= maximum):
            return label
    raise ValueError('Passage must have at least one decision; got {}'.format(size))


def _package_passages(package, novel):
    if package.get('schema_version') != 2 or package.get('novel') != novel:
        raise ValueError('Unsupported or mismatched context package')
    passages = package.get('passages')
    if not isinstance(passages, list):
        raise ValueError('Context package passages must be a list')

    seen_passages = set()
    seen_assignments = set()
    expected_total = 0
    for passage in passages:
        passage_id = passage.get('passage_id')
        assignments = passage.get('assignments')
        expected = passage.get('expected_decision_count')
        if (
            not isinstance(passage_id, str)
            or passage_id in seen_passages
            or not isinstance(assignments, list)
            or not isinstance(expected, int)
            or isinstance(expected, bool)
        ):
            raise ValueError('Invalid or duplicate passage entry in package')
        seen_passages.add(passage_id)
        pending = [
            assignment for assignment in assignments
            if assignment.get('decision_required')
        ]
        if expected != len(pending) or expected < 1:
            raise ValueError(
                'Passage {} declares {} decisions but contains {}'.format(
                    passage_id, expected, len(pending)
                )
            )
        assignment_ids = []
        for assignment in pending:
            assignment_id = assignment.get('assignment_id')
            quote_index = assignment.get('quote_index')
            if (
                not isinstance(assignment_id, str)
                or not isinstance(quote_index, int)
                or isinstance(quote_index, bool)
                or assignment_id != '{}:quote:{}'.format(novel, quote_index)
                or assignment_id in seen_assignments
            ):
                raise ValueError(
                    'Invalid or duplicate assignment in passage {}'.format(
                        passage_id
                    )
                )
            seen_assignments.add(assignment_id)
            assignment_ids.append(assignment_id)
        token_count = passage.get('token_count')
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise ValueError(
                'Passage {} has an invalid token_count'.format(passage_id)
            )
        expected_total += expected
        yield {
            'passage_id': passage_id,
            'expected_decision_count': expected,
            'decision_assignment_ids': assignment_ids,
            'token_count': token_count,
        }

    if package.get('passage_count') != len(seen_passages):
        raise ValueError('Package passage_count does not match passage list')
    if package.get('expected_output_decision_count') != expected_total:
        raise ValueError('Package expected decision total does not match passages')


def _load_results(path, package_bytes, model, novel, passage_records,
                  allowed_character_ids, expected_decision_count):
    if not os.path.isfile(path):
        return None
    results = _read_json(path)
    if results.get('schema_version') != 2:
        raise ValueError('Unsupported Ollama result schema')
    if results.get('decision_policy') != 'forced_choice':
        raise ValueError('Expected forced-choice Ollama results')
    if results.get('source_package_sha256') != hashlib.sha256(
        package_bytes
    ).hexdigest():
        raise ValueError('Ollama results were generated from a different package')
    if results.get('model') != model:
        raise ValueError('Ollama result model does not match {}'.format(model))
    if results.get('novel') != novel:
        raise ValueError('Ollama results novel does not match {}'.format(novel))
    if results.get('expected_passage_count') != len(passage_records):
        raise ValueError('Ollama expected-passage count does not match package')
    if results.get('expected_decision_count') != expected_decision_count:
        raise ValueError('Ollama expected-decision count does not match package')
    if not isinstance(results.get('passages'), dict):
        raise ValueError('Ollama results passages must be an object')
    failed_passages = results.get('failed_passages', {})
    if not isinstance(failed_passages, dict):
        raise ValueError('failed_passages must be an object')

    records_by_id = {record['passage_id']: record for record in passage_records}
    unknown_passages = set(results['passages']) - set(records_by_id)
    if unknown_passages:
        raise ValueError(
            'Results include passages absent from package: {}'.format(
                ', '.join(sorted(unknown_passages))
            )
        )
    for passage_id, result in results['passages'].items():
        record = records_by_id[passage_id]
        if not isinstance(result, dict):
            raise ValueError('Invalid result for passage {}'.format(passage_id))
        decisions = result.get('decisions')
        if (
            result.get('expected_decision_count')
            != record['expected_decision_count']
            or not isinstance(decisions, list)
            or len(decisions) != record['expected_decision_count']
        ):
            raise ValueError(
                'Completed result for {} has an unexpected decision count'.format(
                    passage_id
                )
            )
        decision_ids = [decision.get('assignment_id') for decision in decisions]
        if set(decision_ids) != set(record['decision_assignment_ids']):
            raise ValueError(
                'Completed result assignments differ from package for {}'.format(
                    passage_id
                )
            )
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError(
                    'Invalid decision object for passage {}'.format(passage_id)
                )
            pdnc_id = decision.get('pdncID')
            if (
                decision.get('decision') != 'resolve'
                or not isinstance(pdnc_id, int)
                or isinstance(pdnc_id, bool)
                or pdnc_id not in allowed_character_ids
            ):
                raise ValueError(
                    'Invalid forced-choice decision for {}'.format(passage_id)
                )

    attempts_by_passage = results.get('attempts_by_passage', {})
    if not isinstance(attempts_by_passage, dict):
        raise ValueError('attempts_by_passage must be an object')
    if set(attempts_by_passage) - set(records_by_id):
        raise ValueError('Attempt history contains passages absent from package')

    for record in passage_records:
        passage_id = record['passage_id']
        record['completed'] = passage_id in results['passages']
        record['failed'] = passage_id in results.get('failed_passages', {})
        attempts = attempts_by_passage.get(passage_id, [])
        if not isinstance(attempts, list):
            raise ValueError(
                'Attempt history for {} must be a list'.format(passage_id)
            )
        if any(not isinstance(attempt, dict) for attempt in attempts):
            raise ValueError(
                'Attempt history for {} contains an invalid entry'.format(
                    passage_id
                )
            )
        record['attempt_count'] = len(attempts)
        record['invalid_attempt_count'] = sum(
            attempt.get('status') != 'valid' for attempt in attempts
        )
        record['prompt_tokens'] = sum(
            _usage_value(attempt, 'prompt_eval_count') for attempt in attempts
        )
        record['generated_tokens'] = sum(
            _usage_value(attempt, 'eval_count') for attempt in attempts
        )
        record['valid_prompt_tokens'] = sum(
            _usage_value(attempt, 'prompt_eval_count')
            for attempt in attempts if attempt.get('status') == 'valid'
        )
        record['valid_generated_tokens'] = sum(
            _usage_value(attempt, 'eval_count')
            for attempt in attempts if attempt.get('status') == 'valid'
        )
        record['api_duration_seconds'] = sum(
            _usage_value(attempt, 'total_duration') for attempt in attempts
        ) / 1_000_000_000
    return results


def _usage_value(attempt, field):
    usage = attempt.get('usage', {})
    if not isinstance(usage, dict):
        raise ValueError('Attempt usage must be an object')
    value = usage.get(field, 0)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0
    return value


def _attach_accuracy(passage_records, results, gold_by_quote):
    decisions_by_id = {}
    if results is not None:
        for passage_result in results['passages'].values():
            for decision in passage_result['decisions']:
                decisions_by_id[decision['assignment_id']] = decision

    for record in passage_records:
        comparable_count = 0
        correct_count = 0
        completed_comparable_count = 0
        for assignment_id in record['decision_assignment_ids']:
            quote_index = int(assignment_id.rsplit(':', 1)[1])
            gold_ids = gold_by_quote.get(quote_index, set()) if gold_by_quote else set()
            if not gold_ids:
                continue
            comparable_count += 1
            decision = decisions_by_id.get(assignment_id)
            if decision is not None:
                completed_comparable_count += 1
                correct_count += decision['pdncID'] in gold_ids
        record['gold_comparable_decisions'] = comparable_count
        record['gold_comparable_completed_decisions'] = completed_comparable_count
        record['correct_decisions'] = correct_count


def _summarize_group(records):
    passage_count = len(records)
    decision_count = sum(r['expected_decision_count'] for r in records)
    completed_passages = sum(r.get('completed', False) for r in records)
    completed_decisions = sum(
        r['expected_decision_count'] for r in records if r.get('completed', False)
    )
    comparable = sum(r.get('gold_comparable_decisions', 0) for r in records)
    completed_comparable = sum(
        r.get('gold_comparable_completed_decisions', 0) for r in records
    )
    correct = sum(r.get('correct_decisions', 0) for r in records)
    attempts = sum(r.get('attempt_count', 0) for r in records)
    invalid_attempts = sum(r.get('invalid_attempt_count', 0) for r in records)
    prompt_tokens = sum(r.get('prompt_tokens', 0) for r in records)
    generated_tokens = sum(r.get('generated_tokens', 0) for r in records)
    valid_prompt_tokens = sum(r.get('valid_prompt_tokens', 0) for r in records)
    valid_generated_tokens = sum(r.get('valid_generated_tokens', 0) for r in records)
    total_tokens = prompt_tokens + generated_tokens
    return {
        'passage_count': passage_count,
        'expected_decision_count': decision_count,
        'passage_share': None,
        'decision_share': None,
        'completed_passage_count': completed_passages,
        'passage_completion_rate': (
            completed_passages / passage_count if passage_count else None
        ),
        'completed_decision_count': completed_decisions,
        'decision_completion_rate': (
            completed_decisions / decision_count if decision_count else None
        ),
        'mean_decisions_per_passage': (
            decision_count / passage_count if passage_count else None
        ),
        'max_decisions_per_passage': max(
            (r['expected_decision_count'] for r in records), default=0
        ),
        'mean_context_tokens_per_passage': (
            sum(r['token_count'] for r in records) / passage_count
            if passage_count else None
        ),
        'max_context_tokens_per_passage': max(
            (r['token_count'] for r in records), default=0
        ),
        'generation_attempt_count': attempts,
        'invalid_attempt_count': invalid_attempts,
        'attempts_per_completed_passage': (
            attempts / completed_passages if completed_passages else None
        ),
        'prompt_tokens_all_attempts': prompt_tokens,
        'generated_tokens_all_attempts': generated_tokens,
        'total_tokens_all_attempts': total_tokens,
        'prompt_tokens_valid_attempts': valid_prompt_tokens,
        'generated_tokens_valid_attempts': valid_generated_tokens,
        'total_tokens_valid_attempts': valid_prompt_tokens + valid_generated_tokens,
        'mean_tokens_per_expected_decision': (
            total_tokens / decision_count if decision_count else None
        ),
        'mean_prompt_tokens_per_passage': (
            prompt_tokens / passage_count if passage_count else None
        ),
        'api_duration_seconds_all_attempts': sum(
            r.get('api_duration_seconds', 0) for r in records
        ),
        'gold_comparable_decision_count': comparable,
        'completed_gold_comparable_decision_count': completed_comparable,
        'correct_decision_count': correct,
        'accuracy_when_predicted': (
            correct / completed_comparable if completed_comparable else None
        ),
        'prediction_coverage_on_gold_decisions': (
            completed_comparable / comparable if comparable else None
        ),
        'correct_rate_over_all_gold_decisions': (
            correct / comparable if comparable else None
        ),
    }


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_novel_package_sizes(
    novel, model='qwen2.5:7b', max_decisions_per_passage=None
):
    if novel not in create_book_corefs.novels:
        raise ValueError('Unknown novel: {}'.format(novel))
    novel_output = os.path.join(create_book_corefs.COREF_ROOT, novel)
    package_path = os.path.join(
        novel_output, context_package_filename(max_decisions_per_passage)
    )
    results_path = os.path.join(
        novel_output,
        ollama_results_filename(
            _safe_model_label(model), max_decisions_per_passage
        ),
    )
    quote_report_path = os.path.join(novel_output, 'speaker_assignments_by_quote.csv')
    if not os.path.isfile(package_path):
        raise FileNotFoundError('Context package not found: {}'.format(package_path))
    with open(package_path, 'rb') as package_file:
        package_bytes = package_file.read()
    package = json.loads(package_bytes.decode('utf-8'))
    if package.get('max_decisions_per_passage') != max_decisions_per_passage:
        raise ValueError(
            'Context package decision limit does not match the requested limit'
        )
    records = list(_package_passages(package, novel))
    character_catalog = package.get('character_catalog')
    if not isinstance(character_catalog, list):
        raise ValueError('Context package character_catalog must be a list')
    try:
        allowed_character_ids = {
            int(character['pdncID']) for character in character_catalog
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Context package contains an invalid character catalog') from exc
    expected_decisions = sum(r['expected_decision_count'] for r in records)
    results = _load_results(
        results_path,
        package_bytes,
        model,
        novel,
        records,
        allowed_character_ids,
        expected_decisions,
    )
    gold_by_quote = _read_gold_speakers(quote_report_path)
    _attach_accuracy(records, results, gold_by_quote)

    all_decisions = sum(r['expected_decision_count'] for r in records)
    all_passages = len(records)
    exact_size_counts = Counter(
        r['expected_decision_count'] for r in records
    )
    per_size_records = defaultdict(list)
    for record in records:
        per_size_records[record['expected_decision_count']].append(record)

    size_rows = []
    for size in sorted(per_size_records):
        row = {'decision_count_per_passage': size}
        row.update(_summarize_group(per_size_records[size]))
        row['passage_share'] = row['passage_count'] / all_passages
        row['decision_share'] = row['expected_decision_count'] / all_decisions
        size_rows.append(row)

    bucket_rows = []
    for label, minimum, maximum in SIZE_BUCKETS:
        group = [
            record for record in records
            if record['expected_decision_count'] >= minimum
            and (maximum is None or record['expected_decision_count'] <= maximum)
        ]
        row = {'decision_count_bucket': label}
        row.update(_summarize_group(group))
        row['passage_share'] = row['passage_count'] / all_passages
        row['decision_share'] = row['expected_decision_count'] / all_decisions
        bucket_rows.append(row)

    histogram = [
        {
            'decisions_per_passage': size,
            'passages': exact_size_counts[size],
            'assignments': size * exact_size_counts[size],
            'passage_share': exact_size_counts[size] / all_passages,
            'assignment_share': size * exact_size_counts[size] / all_decisions,
        }
        for size in sorted(exact_size_counts)
    ]
    summary = {
        'schema_version': 1,
        'novel': novel,
        'model': model,
        'max_decisions_per_passage': max_decisions_per_passage,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'context_package': os.path.relpath(package_path),
            'inference_results': (
                os.path.relpath(results_path) if results is not None else None
            ),
            'baseline_quote_report': os.path.relpath(quote_report_path),
        },
        'inference_results_available': results is not None,
        'gold_annotations_available': gold_by_quote is not None,
        'package_totals': {
            'passage_count': all_passages,
            'assignment_count': all_decisions,
            'mean_decisions_per_passage': (
                all_decisions / all_passages if all_passages else 0
            ),
            'median_decisions_per_passage': (
                median(
                    r['expected_decision_count'] for r in records
                )
                if all_passages else None
            ),
            'max_decisions_per_passage': max(
                (r['expected_decision_count'] for r in records), default=0
            ),
            'passages_with_15_or_more_decisions': sum(
                r['expected_decision_count'] >= 15 for r in records
            ),
            'passage_share_15_or_more_decisions': (
                sum(r['expected_decision_count'] >= 15 for r in records)
                / all_passages if all_passages else 0
            ),
            'assignments_in_passages_with_15_or_more_decisions': sum(
                r['expected_decision_count']
                for r in records if r['expected_decision_count'] >= 15
            ),
            'assignment_share_in_passages_with_15_or_more_decisions': (
                sum(
                    r['expected_decision_count']
                    for r in records if r['expected_decision_count'] >= 15
                ) / all_decisions if all_decisions else 0
            ),
            'exact_size_histogram': histogram,
        },
        'bucket_metrics': bucket_rows,
        'outputs': {
            'summary_json': os.path.join(
                novel_output,
                'speaker_assignment_package_size_analysis_{}{}.json'.format(
                    _safe_model_label(model),
                    decision_limit_suffix(max_decisions_per_passage),
                ),
            ),
            'by_size_csv': os.path.join(
                novel_output,
                'speaker_assignment_package_size_analysis_{}{}_by_size.csv'.format(
                    _safe_model_label(model),
                    decision_limit_suffix(max_decisions_per_passage),
                ),
            ),
            'by_passage_csv': os.path.join(
                novel_output,
                'speaker_assignment_package_size_analysis_{}{}_by_passage.csv'.format(
                    _safe_model_label(model),
                    decision_limit_suffix(max_decisions_per_passage),
                ),
            ),
        },
    }
    passage_rows = []
    for record in records:
        row = {
            'novel': novel,
            'passage_id': record['passage_id'],
            'decision_count': record['expected_decision_count'],
            'decision_count_bucket': _bucket_for_size(
                record['expected_decision_count']
            ),
            'context_token_count': record['token_count'],
            'completed': record.get('completed', False),
            'failed': record.get('failed', False),
            'attempt_count': record.get('attempt_count', 0),
            'invalid_attempt_count': record.get('invalid_attempt_count', 0),
            'prompt_tokens_all_attempts': record.get('prompt_tokens', 0),
            'generated_tokens_all_attempts': record.get('generated_tokens', 0),
            'total_tokens_all_attempts': (
                record.get('prompt_tokens', 0)
                + record.get('generated_tokens', 0)
            ),
            'api_duration_seconds_all_attempts': record.get(
                'api_duration_seconds', 0
            ),
            'gold_comparable_decisions': record['gold_comparable_decisions'],
            'completed_gold_comparable_decisions': (
                record['gold_comparable_completed_decisions']
            ),
            'correct_decisions': record['correct_decisions'],
            'accuracy_when_predicted': (
                record['correct_decisions']
                / record['gold_comparable_completed_decisions']
                if record['gold_comparable_completed_decisions'] else None
            ),
            'decision_assignment_ids': json.dumps(
                record['decision_assignment_ids'], ensure_ascii=False
            ),
        }
        passage_rows.append(row)

    with open(summary['outputs']['summary_json'], 'w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)
        output_file.write('\n')
    _write_csv(summary['outputs']['by_size_csv'], size_rows)
    _write_csv(summary['outputs']['by_passage_csv'], passage_rows)

    print(
        '{}: {} passages, {} decisions; average {:.2f}, median {}, maximum {}; '
        '{} passages have >=15 decisions.'.format(
            novel,
            all_passages,
            all_decisions,
            summary['package_totals']['mean_decisions_per_passage'],
            summary['package_totals']['median_decisions_per_passage'],
            summary['package_totals']['max_decisions_per_passage'],
            summary['package_totals']['passages_with_15_or_more_decisions'],
        )
    )
    if results is not None:
        print('Metrics include current inference completion, token, retry, and accuracy data.')
    else:
        print('No matching forced-choice results found; inference metrics are unavailable.')
    for path in summary['outputs'].values():
        print('Wrote {}'.format(path))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Measure grouped speaker-assignment package sizes and their '
            'association with inference cost, completion, and accuracy.'
        )
    )
    parser.add_argument('--novel', required=True, choices=create_book_corefs.novels)
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument(
        '--max-decisions-per-passage',
        type=int,
        help='Analyze the matching capped context package and result files.',
    )
    args = parser.parse_args()
    analyze_novel_package_sizes(
        args.novel,
        model=args.model,
        max_decisions_per_passage=args.max_decisions_per_passage,
    )


if __name__ == '__main__':
    main()
