from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType

def _number(value, name: str, low: float, high: float, *, open_interval: bool=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    valid = low < result < high if open_interval else low <= result <= high
    if not valid:
        raise ValueError(f'{name} outside its permitted range')
    return result

def _budget(value, name: str='budget') -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)

def _item(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError('Item IDs must be nonempty strings')
    return value

def _items(values) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError('Supply a collection of item IDs, not one string')
    return frozenset((_item(value) for value in values))

def _scores(values: Mapping[str, float], name: str='scores') -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f'{name} must be a mapping of item IDs to scores')
    return {_item(item): _number(value, name, 0.0, 1.0) for item, value in values.items()}

def ground_round(catalog_items: Sequence[str], description_cosines: Sequence[Sequence[float]], *, tau: float, v_max: int) -> dict[str, float]:

    tau = _number(tau, 'tau', -1.0, 1.0)
    v_max = _budget(v_max, 'v_max')
    if isinstance(catalog_items, (str, bytes)):
        raise ValueError('Catalog must be an ordered collection of item IDs')
    catalog = tuple((_item(item) for item in catalog_items))
    if not catalog or len(set(catalog)) != len(catalog):
        raise ValueError('Catalog must be nonempty and contain unique item IDs')
    values: dict[str, list[float]] = {}
    for row in description_cosines[:v_max]:
        if len(row) != len(catalog):
            raise ValueError('Cosine row length does not match the catalog')
        cosines = [_number(value, 'cosine', -1.0, 1.0) for value in row]
        index = max(range(len(catalog)), key=cosines.__getitem__)
        if cosines[index] >= tau:
            values.setdefault(catalog[index], []).append((1.0 + cosines[index]) / 2.0)
    return {item: math.fsum(scores) / len(scores) for item, scores in values.items()}

def prefix_scores(round_scores: Sequence[Mapping[str, float]], user_relevance: Mapping[str, float], *, gamma: float) -> tuple[dict[str, float], ...]:

    gamma = _number(gamma, 'gamma', 0.0, 1.0)
    relevance = _scores(user_relevance, 'user relevance')
    matching_rounds: dict[str, list[float]] = {}
    prefixes = []
    for scores in round_scores:
        for item, value in _scores(scores, 'per-round grounding mean').items():
            if item not in relevance:
                raise ValueError(f'Missing user relevance for candidate {item!r}')
            matching_rounds.setdefault(item, []).append(value)
        prefixes.append({item: math.fsum((gamma * (math.fsum(values) / len(values)), (1.0 - gamma) * relevance[item])) for item, values in matching_rounds.items()})
    return tuple(prefixes)

def select_bundle(scores: Mapping[str, float], threshold: float) -> frozenset[str]:

    threshold = _number(threshold, 'threshold', 0.0, 1.0)
    return frozenset((item for item, score in _scores(scores).items() if score >= threshold))

def fbeta_loss(predicted, truth, beta: float=1.0) -> float:

    beta = _number(beta, 'beta', 0.0, math.inf, open_interval=True)
    predicted, truth = (_items(predicted), _items(truth))
    if not predicted and (not truth):
        return 0.0
    hits = len(predicted & truth)
    if hits == 0:
        return 1.0
    if beta >= 1.0:
        inverse_squared = (1.0 / beta) ** 2
        prediction_weight = inverse_squared / (1.0 + inverse_squared)
        truth_weight = 1.0 / (1.0 + inverse_squared)
    else:
        squared = beta * beta
        prediction_weight = 1.0 / (1.0 + squared)
        truth_weight = squared / (1.0 + squared)
    denominator = math.fsum((prediction_weight * len(predicted), truth_weight * len(truth)))
    return min(1.0, max(0.0, 1.0 - hits / denominator))

def mean_loss(predictions: Sequence, truths: Sequence, beta: float=1.0) -> float:

    if not predictions or len(predictions) != len(truths):
        raise ValueError('Predictions/truths must have the same positive length')
    return math.fsum((fbeta_loss(pred, truth, beta) for pred, truth in zip(predictions, truths))) / len(predictions)

def bernoulli_kl(p: float, q: float) -> float:

    p, q = (_number(p, 'p', 0.0, 1.0), _number(q, 'q', 0.0, 1.0))
    if p == q:
        return 0.0
    if q in (0.0, 1.0):
        return math.inf
    if p == 0.0:
        return -math.log1p(-q)
    if p == 1.0:
        return -math.log(q)
    terms = (p * (math.log(p) - math.log(q)), (1.0 - p) * (math.log1p(-p) - math.log1p(-q)))
    return max(0.0, math.fsum(terms))

def log_risk_pvalue(empirical_risk: float, alpha: float, n_cal: int) -> float:

    risk = _number(empirical_risk, 'empirical risk', 0.0, 1.0)
    alpha = _number(alpha, 'alpha', 0.0, 1.0, open_interval=True)
    n_cal = _budget(n_cal, 'n_cal')
    return -n_cal * bernoulli_kl(risk, alpha) if risk < alpha else 0.0

def risk_pvalue(empirical_risk: float, alpha: float, n_cal: int) -> float:
    return math.exp(log_risk_pvalue(empirical_risk, alpha, n_cal))

@dataclass(frozen=True)
class CalibrationGrid:

    budgets: tuple[int, ...]
    thresholds: tuple[float, ...]

    def __post_init__(self):
        budgets = tuple((_budget(s) for s in self.budgets))
        thresholds = tuple((_number(t, 'threshold', 0.0, 1.0) for t in self.thresholds))
        if not budgets or not thresholds:
            raise ValueError('Calibration grids must be nonempty')
        if len(set(budgets)) != len(budgets) or len(set(thresholds)) != len(thresholds):
            raise ValueError('Calibration grids must not contain duplicate entries')
        object.__setattr__(self, 'budgets', tuple(sorted(budgets)))
        object.__setattr__(self, 'thresholds', tuple(sorted(thresholds)))

    @property
    def size(self) -> int:
        return len(self.budgets) * len(self.thresholds)

@dataclass(frozen=True)
class CalibrationExample:

    scores_by_budget: Mapping[int, Mapping[str, float]]
    truth: frozenset[str]

    def __post_init__(self):
        if not isinstance(self.scores_by_budget, Mapping):
            raise ValueError('scores_by_budget must be a mapping')
        frozen = {_budget(s): MappingProxyType(_scores(scores)) for s, scores in self.scores_by_budget.items()}
        object.__setattr__(self, 'scores_by_budget', MappingProxyType(frozen))
        object.__setattr__(self, 'truth', _items(self.truth))

    @classmethod
    def from_rounds(cls, round_scores, user_relevance, *, gamma: float, truth):
        prefixes = prefix_scores(round_scores, user_relevance, gamma=gamma)
        return cls({s: scores for s, scores in enumerate(prefixes, 1)}, _items(truth))

@dataclass(frozen=True)
class PairEvaluation:
    budget: int
    threshold: float
    empirical_risk: float
    p_value: float
    log_p_value: float
    certified: bool

@dataclass(frozen=True)
class CalibrationResult:
    status: str
    selected: PairEvaluation | None
    evaluations: tuple[PairEvaluation, ...]
    grid: CalibrationGrid
    n_cal: int
    alpha: float
    delta: float
    beta: float
    bonferroni_cutoff: float
    log_bonferroni_cutoff: float

def calibrate(examples: Sequence[CalibrationExample], grid: CalibrationGrid, *, alpha: float, delta: float, beta: float=1.0) -> CalibrationResult:

    if not isinstance(grid, CalibrationGrid):
        raise ValueError('Supply an explicit CalibrationGrid')
    examples = tuple(examples)
    if not examples or not all((isinstance(e, CalibrationExample) for e in examples)):
        raise ValueError('Supply at least one CalibrationExample')
    alpha = _number(alpha, 'alpha', 0.0, 1.0, open_interval=True)
    delta = _number(delta, 'delta', 0.0, 1.0, open_interval=True)
    beta = _number(beta, 'beta', 0.0, math.inf, open_interval=True)
    for example in examples:
        if any((s not in example.scores_by_budget for s in grid.budgets)):
            raise ValueError('Each example must supply scores for every declared budget')
    log_cutoff = math.log(delta) - math.log(grid.size)
    evaluations = []
    for budget in grid.budgets:
        for threshold in grid.thresholds:
            risk = math.fsum((fbeta_loss(select_bundle(example.scores_by_budget[budget], threshold), example.truth, beta) for example in examples)) / len(examples)
            log_p = log_risk_pvalue(risk, alpha, len(examples))
            evaluations.append(PairEvaluation(budget, threshold, risk, math.exp(log_p), log_p, log_p <= log_cutoff))
    certified = [pair for pair in evaluations if pair.certified]
    selected = min(certified, key=lambda pair: (pair.budget, -pair.threshold)) if certified else None
    return CalibrationResult(status='certified' if selected else 'no_certified_pair', selected=selected, evaluations=tuple(evaluations), grid=grid, n_cal=len(examples), alpha=alpha, delta=delta, beta=beta, bonferroni_cutoff=delta / grid.size, log_bonferroni_cutoff=log_cutoff)
