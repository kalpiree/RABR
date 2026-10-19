import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from rabr.data import atomic_json, canonical, digest, load_dataset, read_jsonl
from rabr.metrics import evaluate_records
from rabr.queue import contained, read_manifest, validate_receipt


def group_configuration(root, job, jobs):
    config = json.loads(contained(root, job['config']).read_text())
    algorithm = config.get('algorithm', {})
    if 'rabr_reference' in algorithm:
        reference_path = (contained(root, job['output']) / algorithm['rabr_reference']).resolve()
        references = [other for other in jobs if other['id'] in job['requires']
                      and (contained(root, other['output']) / 'result.json').resolve() == reference_path]
        if len(references) != 1:
            raise ValueError('Calibration reference does not match one declared dependency')
        reference = references[0]
        if reference['method'] != 'rabr' or reference['domain'] != job['domain'] or reference['seed'] != job['seed']:
            raise ValueError('Calibration reference identity differs')
        algorithm['rabr_reference'] = {key: reference[key] for key in
                                      ('method', 'backbone', 'domain', 'config_sha256', 'data_manifest_sha256')}
    return digest(config)


def aggregate(queue_path):
    queue_path = Path(queue_path).resolve()
    manifest = read_manifest(queue_path)
    groups, runs = defaultdict(list), []
    seen = set()
    code_hashes = set()
    datasets = {}
    for job in manifest['jobs']:
        result = validate_receipt(queue_path.parent, job)
        if result is None:
            raise ValueError('Incomplete job: ' + job['id'])
        key = canonical({'domain': job['domain'], 'method': job['method'], 'backbone': job['backbone'],
                         'research': job['research'],
                         'group_config_sha256': group_configuration(queue_path.parent, job, manifest['jobs'])}).decode()
        identity = (key, job['seed'])
        if identity in seen:
            raise ValueError('Duplicate configuration and seed')
        seen.add(identity)
        output = contained(queue_path.parent, job['output'])
        metrics = json.loads((output / 'metrics.json').read_text()) if result['status'] == 'complete' else None
        if result['status'] == 'complete':
            if job['domain'] not in datasets:
                datasets[job['domain']] = load_dataset((queue_path.parent / manifest['data']).resolve(), job['domain'])
            targets = {row['session_id']: row['target']['gold_items'] for row in datasets[job['domain']].test}
            predictions = read_jsonl(output / 'predictions.jsonl')
            if len(predictions) != len(targets) or {row['session_id'] for row in predictions} != set(targets):
                raise ValueError('Predictions must cover each test session exactly once')
            checked, per_session = evaluate_records([{**row, 'gold_items': targets[row['session_id']]} for row in predictions], beta=metrics['beta'])
            if checked != metrics or read_jsonl(output / 'per-session.jsonl') != per_session:
                raise ValueError('Metric artifacts differ from predictions and test labels')
        cost = json.loads((output / 'cost.json').read_text()) if 'cost.json' in result['artifacts'] else None
        row = {'job_id': job['id'], 'seed': job['seed'], 'status': result['status'], 'metrics': metrics, 'cost': cost,
               'config_sha256': result['config_sha256'], 'data_manifest_sha256': result['data_manifest_sha256']}
        code_hashes.add(result['code_sha256'])
        groups[key].append(row)
        runs.append(row)
    if len(code_hashes) != 1:
        raise ValueError('Cannot aggregate runs produced by different source code')
    summaries = []
    for key, values in groups.items():
        description = json.loads(key)
        successful = [row for row in values if row['status'] == 'complete']
        fields = ('macro_precision', 'macro_recall', 'hit_ratio', 'macro_f_beta', 'mean_risk', 'mean_bundle_size')
        metrics = {}
        for field in fields:
            numbers = [row['metrics'][field] for row in successful]
            if any(type(value) not in (float, int) or not math.isfinite(value) for value in numbers):
                raise ValueError('Non-finite metric')
            metrics[field] = {'mean': statistics.mean(numbers) if numbers else None,
                              'sample_sd': statistics.stdev(numbers) if len(numbers) > 1 else None,
                              'completed_seeds': len(numbers)}
        counts = {row['metrics']['session_count'] for row in successful}
        if len(counts) > 1:
            raise ValueError('Test session count differs across seeds')
        summaries.append({**description, 'seeds': sorted(row['seed'] for row in values),
                          'completed_seeds': len(successful), 'no_certified_pair_seeds': len(values)-len(successful),
                          'unique_test_sessions': next(iter(counts)) if counts else None, 'metrics': metrics})
    return {'schema_version': 1, 'profile': manifest['profile'], 'code_sha256': next(iter(code_hashes)),
            'interpretation': 'fresh_protocol_seed_descriptives_not_original_manuscript_tables',
            'uncertainty': 'sample_sd_across_seeds_on_the_same_test_sessions_not_independent_examples',
            'grouping': 'exact_configuration_with_calibration_reference_path_replaced_by_dependency_semantics',
            'cost_scope': 'per_run_recorded_costs_preserved_no_claim_to_include_uncommitted_interrupted_work',
            'jobs': len(runs), 'groups': summaries, 'runs': runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = aggregate(args.queue)
    atomic_json(args.output, result)
    print(json.dumps({'jobs': result['jobs'], 'groups': len(result['groups'])}))


if __name__ == '__main__':
    main()
