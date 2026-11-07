def decision_limit_suffix(max_decisions_per_passage):
    if max_decisions_per_passage is None:
        return ''
    if max_decisions_per_passage < 1:
        raise ValueError('max_decisions_per_passage must be positive')
    return '_max{}'.format(max_decisions_per_passage)


def split_passage_group(group, max_decisions_per_passage):
    if max_decisions_per_passage is None:
        return [group]
    if max_decisions_per_passage < 1:
        raise ValueError('max_decisions_per_passage must be positive')
    rows = group['rows']
    chunks = [
        rows[index:index + max_decisions_per_passage]
        for index in range(0, len(rows), max_decisions_per_passage)
    ]
    return [
        {
            'rows': chunk,
            'context_start': group['context_start'],
            'context_end': group['context_end'],
            'part_number': part_number,
            'part_count': len(chunks),
        }
        for part_number, chunk in enumerate(chunks, start=1)
    ]


def context_package_filename(max_decisions_per_passage=None):
    return 'speaker_assignment_llm_context_packages{}.json'.format(
        decision_limit_suffix(max_decisions_per_passage)
    )


def context_summary_filename(max_decisions_per_passage=None):
    return 'speaker_assignment_llm_context_summary{}.json'.format(
        decision_limit_suffix(max_decisions_per_passage)
    )


def ollama_results_filename(model_label, max_decisions_per_passage=None):
    return 'speaker_assignment_llm_ollama_results_{}_forced_choice{}.json'.format(
        model_label,
        decision_limit_suffix(max_decisions_per_passage),
    )
