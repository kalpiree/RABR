import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from rabr.data import atomic_json, canonical, digest, load_dataset, read_jsonl
from rabr.metrics import evaluate_records
from rabr.queue import contained, read_manifest, validate_receipt
from rabr.statistics import holm_adjust, paired_test

PAPER_BASELINES = {'pog', 'bunt', 'bgn', 'byob', 'zero_shot', 'llm4b', 'text2bundle', 'aicl'}
RECENT_BASELINES = {'kd4bg', 'ddbc'}


def evaluation_condition(config):
    algorithm = {**config.get('algorithm', {}), **config.get('research', {}).get('parameters', {})}
    if config['method'] == 'bunt':
        return 'feedback_assisted_full_bundle'
    if config['method'] == 'ddbc':
        context = algorithm.get('context_mode', 'provided_seed_items')
        output = algorithm.get('output_mode', 'completion_only')
        if context == 'provided_seed_items':
            return 'supplied_seed_completion' if output == 'completion_only' else 'supplied_seed_full_bundle'
        if output == 'completion_only':
            return 'retrieved_seed_completion'
    return 'query_history_full_bundle'


def paired_comparisons(groups, *, exact_permutation=False, alpha=.05):
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('Significance level must lie strictly between zero and one')
    descriptions = {key: json.loads(key) for key in groups}
    rows, excluded = [], []
    scopes = {'manuscript_baselines': PAPER_BASELINES, 'including_recent_baselines': PAPER_BASELINES | RECENT_BASELINES}
    for reference_key, reference in descriptions.items():
        if reference['method'] != 'rabr' or reference['research']['rq'] != 'rq1' or reference['research']['variant'] != 'main':
            continue
        reference_runs = {row['seed']: row for row in groups[reference_key]}
        candidates = []
        for key, candidate in descriptions.items():
            if candidate['method'] == 'rabr' or candidate['domain'] != reference['domain'] or candidate['research']['rq'] != 'rq1':
                continue
            reference_backbone = candidate.get('reference_backbone', candidate['research'].get('parameters', {}).get('reference_backbone'))
            comparable = candidate['backbone'] == reference['backbone'] or (
                candidate['backbone'] == 'none' and reference_backbone in (None, reference['backbone']))
            if comparable:
                reference_condition = reference.get('evaluation_condition', 'query_history_full_bundle')
                candidate_condition = candidate.get('evaluation_condition', evaluation_condition(candidate))
                if candidate_condition != reference_condition:
                    excluded.append({'domain': reference['domain'], 'backbone': reference['backbone'],
                                     'rabr_group': reference['group_config_sha256'],
                                     'baseline_group': candidate['group_config_sha256'],
                                     'baseline_method': candidate['method'],
                                     'reference_condition': reference_condition,
                                     'baseline_condition': candidate_condition,
                                     'reason': 'different_inputs_feedback_or_evaluation_targets'})
                    continue
                candidates.append((key, candidate))
        for scope, methods in scopes.items():
            eligible = [(key, candidate) for key, candidate in candidates if candidate['method'] in methods]
            present = sorted({candidate['method'] for _, candidate in eligible})
            for metric in ('macro_precision', 'macro_recall', 'hit_ratio'):
                comparison_rows = []
                for baseline_key, baseline in eligible:
                    baseline_runs = {row['seed']: row for row in groups[baseline_key]}
                    record = {'scope': scope, 'domain': reference['domain'], 'backbone': reference['backbone'], 'metric': metric,
                              'rabr_group': reference['group_config_sha256'], 'baseline_group': baseline['group_config_sha256'],
                              'baseline_method': baseline['method'], 'baseline_variant': baseline['research']['variant'],
                              'present_baseline_methods': present, 'complete_method_family': set(present) == methods,
                              'expected_seeds': sorted(reference_runs), 'observed_best_present_competitor': False}
                    if set(reference_runs) != set(baseline_runs) or any(
                        row['status'] != 'complete' for row in [*reference_runs.values(), *baseline_runs.values()]):
                        record.update(status='unavailable_incomplete_seed_results', paired_t=None, baseline_mean=None)
                    elif any(reference_runs[seed]['data_manifest_sha256'] != baseline_runs[seed]['data_manifest_sha256']
                             for seed in reference_runs):
                        raise ValueError('Paired runs use different datasets')
                    else:
                        left = {seed: row['metrics'][metric] for seed, row in reference_runs.items()}
                        right = {seed: row['metrics'][metric] for seed, row in baseline_runs.items()}
                        record.update(status='computed', baseline_mean=statistics.mean(right.values()),
                                      paired_t=paired_test(left, right, alpha=alpha))
                        if exact_permutation:
                            record['exact_sign_flip'] = paired_test(left, right, method='exact_sign_flip', alpha=alpha)
                    comparison_rows.append(record)
                available = [row for row in comparison_rows if row['baseline_mean'] is not None]
                if available:
                    best = min(available, key=lambda row: (-row['baseline_mean'], row['baseline_method'], row['baseline_group']))
                    best['observed_best_present_competitor'] = True
                rows.extend(comparison_rows)
    for scope in scopes:
        family = [row for row in rows if row['scope'] == scope]
        adjusted = holm_adjust([row['paired_t']['p_value'] if row.get('paired_t') and row['paired_t']['p_value'] is not None else 1.0
                               for row in family])
        for row, probability in zip(family, adjusted):
            tested = row.get('paired_t') and row['paired_t']['p_value'] is not None
            row['holm_adjusted_p_value'] = probability if tested else None
            row['holm_reject_equal_means'] = bool(tested and probability < alpha)
            row['holm_family_size'] = len(family)
    return {'primary_test': 'paired_two_sided_t', 'paired_unit': 'seed_level_test_set_metric',
            'nominal_alpha': alpha, 'best_competitor_selection': 'highest_observed_mean_per_metric_with_lexicographic_tie_break',
            'multiplicity': 'Holm_across_all_planned_domain_backbone_metric_baseline_comparisons_within_each_scope',
            'excluded_incompatible_conditions': excluded, 'comparisons': rows}


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
        if 'rabr_reference_identity' in algorithm:
            expected = {key: reference[key] for key in ('method', 'backbone', 'domain', 'seed', 'data_manifest_sha256', 'config_sha256')}
            if algorithm['rabr_reference_identity'] != expected:
                raise ValueError('Calibration reference provenance differs')
            algorithm['rabr_reference_identity'] = algorithm['rabr_reference']
    return digest(config)


def aggregate(queue_path, *, exact_permutation=False, significance_level=.05):
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
        config = json.loads(contained(queue_path.parent, job['config']).read_text())
        key = canonical({'domain': job['domain'], 'method': job['method'], 'backbone': job['backbone'],
                         'reference_backbone': job.get('reference_backbone'),
                         'evaluation_condition': evaluation_condition(config),
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
            if evaluation_condition(config) == 'supplied_seed_completion':
                for example in datasets[job['domain']].test:
                    seeds = example['input'].get('seed_items')
                    if not seeds or set(seeds) & set(example['target']['gold_items']):
                        raise ValueError('Supplied-seed completion requires distinct withheld targets excluding its seeds')
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
            'jobs': len(runs), 'groups': summaries, 'runs': runs,
            'significance': paired_comparisons(groups, exact_permutation=exact_permutation, alpha=significance_level)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--exact-permutation', action='store_true')
    parser.add_argument('--significance-level', type=float, default=.05)
    args = parser.parse_args()
    result = aggregate(args.queue, exact_permutation=args.exact_permutation, significance_level=args.significance_level)
    atomic_json(args.output, result)
    print(json.dumps({'jobs': result['jobs'], 'groups': len(result['groups'])}))


if __name__ == '__main__':
    main()
