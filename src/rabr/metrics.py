import argparse
import json
import math
from pathlib import Path
import sys

def _id_key(value, field):
    if type(value) is int:
        return (int, value)
    if type(value) is str and value:
        return (str, value)
    raise ValueError(f'{field} must be a nonempty string or integer (not boolean)')

def _item_set(value, field):
    if not isinstance(value, list):
        raise ValueError(f'{field} must be a list')
    return {_id_key(item, f'{field}[{i}]') for i, item in enumerate(value)}

def _validated_beta(beta):
    if isinstance(beta, bool):
        raise ValueError('beta must be finite and positive')
    try:
        beta = float(beta)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('beta must be finite and positive') from error
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    return beta

def _f_beta(tp, gold_count, predicted_count, beta):
    if tp == 0:
        return 0.0
    if beta >= 1:
        inverse_squared = (1 / beta) ** 2
        return (1 + inverse_squared) * tp / (gold_count + inverse_squared * predicted_count)
    squared = beta ** 2
    return (1 + squared) * tp / (predicted_count + squared * gold_count)

def evaluate_records(records, beta=1.0):
    """Return (summary, per_session_rows); reject ambiguous or invalid inputs."""
    beta = _validated_beta(beta)
    seen = set()
    per_session = []
    for position, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f'record {position} must be a JSON object')
        missing = {'session_id', 'gold_items', 'predicted_items'} - record.keys()
        if missing:
            raise ValueError(f"record {position} missing fields: {', '.join(sorted(missing))}")
        session_id = record['session_id']
        session_key = _id_key(session_id, f'record {position} session_id')
        if session_key in seen:
            raise ValueError(f'duplicate session_id at record {position}: {session_id!r}')
        seen.add(session_key)
        gold = _item_set(record['gold_items'], f'record {position} gold_items')
        predicted = _item_set(record['predicted_items'], f'record {position} predicted_items')
        if not gold:
            raise ValueError(f'record {position} has empty gold_items; empty gold is unsupported')
        tp = len(gold & predicted)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(gold)
        f_beta = _f_beta(tp, len(gold), len(predicted), beta)
        per_session.append({'session_id': session_id, 'gold_input_count': len(record['gold_items']), 'predicted_input_count': len(record['predicted_items']), 'gold_count': len(gold), 'predicted_count': len(predicted), 'gold_duplicates_removed': len(record['gold_items']) - len(gold), 'predicted_duplicates_removed': len(record['predicted_items']) - len(predicted), 'true_positive_count': tp, 'false_positive_count': len(predicted) - tp, 'false_negative_count': len(gold) - tp, 'precision': precision, 'recall': recall, 'hit_ratio': int(tp > 0), 'f_beta': f_beta, 'risk': 1 - f_beta})
    if not per_session:
        raise ValueError('input contains no sessions')
    n = len(per_session)
    metrics = {'macro_precision': 'precision', 'macro_recall': 'recall', 'hit_ratio': 'hit_ratio', 'macro_f_beta': 'f_beta', 'mean_risk': 'risk', 'mean_bundle_size': 'predicted_count'}
    counts = ['gold_input_count', 'predicted_input_count', 'gold_count', 'predicted_count', 'gold_duplicates_removed', 'predicted_duplicates_removed', 'true_positive_count', 'false_positive_count', 'false_negative_count']
    summary = {'schema_version': 1, 'evaluation': 'macro_item_sets', 'beta': beta, 'empty_gold_policy': 'error', 'item_duplicate_policy': 'deduplicate_with_counts', 'item_id_policy': 'typed_string_or_integer_exact_match', 'risk_interpretation': 'observed_mean_1_minus_f_beta_not_a_certification', 'session_count': n, **{name: math.fsum((row[field] for row in per_session)) / n for name, field in metrics.items()}, 'counts': {**{field: sum((row[field] for row in per_session)) for field in counts}, 'hit_session_count': sum((row['hit_ratio'] for row in per_session)), 'empty_prediction_session_count': sum((row['predicted_count'] == 0 for row in per_session))}}
    return (summary, per_session)

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'invalid JSON on line {line_number}: {error.msg}') from error

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSONL with explicit gold and predicted item IDs')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--output', type=Path, help='summary JSON; default is stdout')
    parser.add_argument('--per-session-output', type=Path, help='optional per-session JSONL')
    args = parser.parse_args(argv)
    try:
        paths = [path.resolve() for path in (args.input, args.output, args.per_session_output) if path]
        if len(set(paths)) != len(paths):
            raise ValueError('input and output paths must be different')
        summary, per_session = evaluate_records(read_jsonl(args.input), beta=args.beta)
        summary_text = json.dumps(summary, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(summary_text, encoding='utf-8')
        else:
            sys.stdout.write(summary_text)
        if args.per_session_output:
            args.per_session_output.write_text(''.join((json.dumps(row, allow_nan=False) + '\n' for row in per_session)), encoding='utf-8')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
