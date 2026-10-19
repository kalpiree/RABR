from __future__ import annotations

import itertools
import math


def _beta_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1 / d
    value = d
    for index in range(1, 10001):
        twice = 2 * index
        aa = index * (b - index) * x / ((qam + twice) * (a + twice))
        d = 1 + aa * d
        c = 1 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        value *= d * c
        aa = -(a + index) * (qab + index) * x / ((a + twice) * (qap + twice))
        d = 1 + aa * d
        c = 1 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        change = d * c
        value *= change
        if abs(change - 1) <= 3e-14:
            return value
    raise ArithmeticError('Incomplete beta continued fraction did not converge')


def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    scale = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        value = scale * _beta_fraction(a, b, x) / a
    else:
        value = 1 - scale * _beta_fraction(b, a, 1 - x) / b
    return max(0.0, min(1.0, value))


def paired_test(left, right, *, method='paired_t', alpha=.05):
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        raise ValueError('Paired tests require exactly matching seed IDs')
    if any(type(seed) is not int for seed in left):
        raise ValueError('Seed IDs must be integers')
    if method not in ('paired_t', 'exact_sign_flip') or not 0 < alpha < 1:
        raise ValueError('Invalid test method or significance level')
    seeds = sorted(left)
    if any(type(value) not in (int, float) or not math.isfinite(value) for values in (left, right) for value in values.values()):
        raise ValueError('Paired scores must be finite numeric values')
    differences = [left[seed] - right[seed] for seed in seeds]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError('Paired differences overflow')
    count = len(seeds)
    mean = math.fsum(differences) / count if count else None
    result = {'test': method, 'alternative': 'two-sided', 'unit': 'matched_seed_level_metric',
              'seeds': seeds, 'pairs': count, 'mean_difference': mean, 'alpha': alpha,
              'statistic': None, 'p_value': None, 'reject_equal_means': False}
    if count < 2:
        return {**result, 'status': 'insufficient_pairs'}
    if method == 'exact_sign_flip':
        if count > 20:
            raise ValueError('Exact sign-flip enumeration is limited to 20 pairs')
        observed = abs(math.fsum(differences))
        tolerance = 16 * math.ulp(max(1.0, math.fsum(abs(value) for value in differences)))
        extreme = sum(abs(math.fsum(sign * value for sign, value in zip(signs, differences))) >= observed - tolerance
                      for signs in itertools.product((-1, 1), repeat=count))
        probability = extreme / (2 ** count)
        return {**result, 'status': 'ok', 'statistic': abs(mean), 'p_value': probability,
                'reject_equal_means': probability < alpha, 'permutations': 2 ** count,
                'assumptions': 'exchangeable_signs_of_paired_differences_under_the_null'}
    variance = math.fsum((value - mean) ** 2 for value in differences) / (count - 1)
    result.update(degrees_of_freedom=count - 1,
                  assumptions='independent_seed_differences_with_approximately_normal_sampling_distribution')
    if variance == 0:
        if mean == 0:
            return {**result, 'status': 'identical_scores', 'statistic': 0.0, 'p_value': 1.0}
        return {**result, 'status': 'undefined_zero_variance_nonzero_difference'}
    statistic = mean / math.sqrt(variance / count)
    degrees = count - 1
    probability = _regularized_beta(degrees / (degrees + statistic * statistic), degrees / 2, .5)
    return {**result, 'status': 'ok', 'statistic': statistic, 'p_value': probability,
            'reject_equal_means': probability < alpha}


def holm_adjust(probabilities):
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError('Expected finite probabilities in [0,1]')
    adjusted = [None] * len(probabilities)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(probabilities)), key=lambda index: (probabilities[index], index))):
        running = max(running, (len(probabilities) - rank) * probabilities[index])
        adjusted[index] = min(1.0, running)
    return adjusted
