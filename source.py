"""Core algorithms for shared multi-semivalue ranking.

The main estimators are:

- IncRank: include-side ranking proxy
- ExcRank: exclude-side ranking proxy
- StdRank: standardized fixed-weight combination
- AdaRank: split-half stability-weighted combination

The baseline estimators are OFA, OFA-S, GELS, WPERM, WSL, WSHAP, and SHAPIQ.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEMIVALUE_SET = (
    "BT-1-1",
    "BT-1-4",
    "BT-4-1",
    "BT-4-4",
    "WBZ-01",
    "WBZ-03",
    "WBZ-05",
    "WBZ-07",
    "WBZ-09",
)


def semivalue_to_param(name: str) -> dict:
    """Convert a paper-facing semivalue name into parameters."""
    if name == "SV":
        return {"kind": "SV"}
    if name == "BZ":
        return {"kind": "BZ"}
    if name.startswith("BT-"):
        _, alpha, beta = name.split("-")
        return {"kind": "BT", "alpha": float(alpha), "beta": float(beta)}
    if name.startswith("WBZ-"):
        _, q = name.split("-")
        return {"kind": "WBZ", "q": int(q) / 10.0}
    raise ValueError(f"Unknown semivalue: {name}")


def semivalue_from_param(param: dict) -> str:
    kind = param["kind"]
    if kind in {"SV", "BZ"}:
        return kind
    if kind == "BT":
        return f"BT-{int(param['alpha'])}-{int(param['beta'])}"
    if kind == "WBZ":
        return f"WBZ-{int(round(float(param['q']) * 10)):02d}"
    raise ValueError(f"Unknown semivalue parameters: {param}")


def _log_comb(n: int, k) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    return special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)


def _normalize_prob(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0) or not np.all(np.isfinite(x)):
        raise ValueError(f"Invalid probability scores: {x}")
    total = float(np.sum(x))
    if total <= 0:
        raise ValueError("Probability scores must have positive total mass.")
    return x / total


def _normalize_log_prob(log_x: np.ndarray) -> np.ndarray:
    log_x = np.asarray(log_x, dtype=np.float64)
    offset = float(np.max(log_x))
    return _normalize_prob(np.exp(log_x - offset))


def semivalue_probabilities(n_players: int, semivalue: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return p_s and log p_s for s=1,...,n."""
    s = np.arange(1, n_players + 1, dtype=np.float64)
    kind = semivalue["kind"]

    if kind == "SV":
        log_p = special.betaln(n_players - s + 1.0, s)
    elif kind == "BZ":
        log_p = np.full(n_players, -(n_players - 1) * np.log(2.0))
    elif kind == "WBZ":
        q = float(semivalue["q"])
        if not 0.0 < q < 1.0:
            raise ValueError(f"Weighted Banzhaf q must be in (0, 1), got {q}.")
        log_p = (s - 1.0) * np.log(q) + (n_players - s) * np.log1p(-q)
    elif kind == "BT":
        alpha = float(semivalue["alpha"])
        beta = float(semivalue["beta"])
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta-Shapley alpha and beta must be positive.")
        log_p = special.betaln(s + beta - 1.0, n_players - s + alpha)
        log_p -= special.betaln(alpha, beta)
    else:
        raise ValueError(f"Unknown semivalue kind: {kind}")

    return np.exp(log_p).astype(float), log_p.astype(float)


@dataclass(frozen=True)
class SemivalueCoefficients:
    p: np.ndarray
    cardinality: np.ndarray
    include_coeff: np.ndarray
    include_constant_coeff: np.ndarray
    exclude_coeff: np.ndarray
    exclude_constant_coeff: np.ndarray
    ofa_include_coeff: np.ndarray
    ofa_exclude_coeff: np.ndarray


def semivalue_coefficients(n_players: int, semivalue: dict) -> SemivalueCoefficients:
    """Coefficients used by the paper estimators."""
    p, log_p = semivalue_probabilities(n_players, semivalue)
    s = np.arange(1, n_players + 1, dtype=np.float64)
    cardinality = np.exp(log_p + _log_comb(n_players - 1, s - 1))

    r = np.arange(1, n_players, dtype=np.float64)
    log_pair = np.logaddexp(log_p[: n_players - 1], log_p[1:])
    include_coeff = np.exp(log_pair + _log_comb(n_players - 1, r - 1))
    include_constant_coeff = np.exp(log_p[1:] + _log_comb(n_players, r))
    exclude_coeff = np.exp(log_pair + _log_comb(n_players - 1, r))
    exclude_constant_coeff = np.exp(log_p[: n_players - 1] + _log_comb(n_players, r))

    ofa_include_coeff = p[: n_players - 1] * special.comb(
        n_players - 1, np.arange(0, n_players - 1), exact=False
    )
    ofa_exclude_coeff = p[1:] * special.comb(
        n_players - 1, np.arange(1, n_players), exact=False
    )

    return SemivalueCoefficients(
        p=p,
        cardinality=cardinality,
        include_coeff=include_coeff,
        include_constant_coeff=include_constant_coeff,
        exclude_coeff=exclude_coeff,
        exclude_constant_coeff=exclude_constant_coeff,
        ofa_include_coeff=ofa_include_coeff,
        ofa_exclude_coeff=ofa_exclude_coeff,
    )


def unanimity_semivalue_coeff(n_players: int, semivalue: str | dict, set_size: int) -> float:
    """Semivalue assigned to each member of a unanimity component."""
    param = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue
    if param["kind"] == "WBZ":
        return float(param["q"] ** (set_size - 1))
    _, log_p = semivalue_probabilities(n_players, param)
    k = np.arange(0, n_players - set_size + 1, dtype=float)
    log_terms = _log_comb(n_players - set_size, k)
    log_terms += log_p[(set_size - 1) + k.astype(int)]
    return float(np.exp(special.logsumexp(log_terms)))


def coalition_size_distribution(
    n_players: int,
    name: str = "ofaa",
    semivalues: Iterable[dict] | None = None,
    alpha: float = 0.0,
    low: int = 1,
    high: int | None = None,
) -> np.ndarray:
    """Distribution over coalition sizes 1,...,n-1."""
    name = name.lower()
    if high is None:
        high = n_players
    sizes = np.arange(low, high, dtype=np.float64)
    if len(sizes) == 0:
        raise ValueError(f"Empty coalition-size support: low={low}, high={high}.")
    if name == "uniform":
        return _normalize_prob(np.ones_like(sizes))
    if name == "poly":
        return _normalize_log_prob(float(alpha) * np.log(sizes * (n_players - sizes)))
    if name in {"ofaa", "ofa-a"}:
        return _normalize_prob(1.0 / np.sqrt(sizes * (n_players - sizes)))
    if name in {"ofas", "ofa-s"}:
        semivalue_list = list(semivalues or [])
        if len(semivalue_list) != 1:
            raise ValueError("OFA-S requires exactly one target semivalue.")
        semivalue_iter = semivalue_list
    elif name in {"ofaset", "ofa-set"}:
        semivalue_iter = list(semivalues or [{"kind": "SV"}])
    else:
        semivalue_iter = None
    if semivalue_iter is not None:
        scores = np.zeros(len(sizes), dtype=float)
        for semivalue in semivalue_iter:
            coeffs = semivalue_coefficients(n_players, semivalue)
            idx = sizes.astype(int) - 1
            left = coeffs.cardinality[idx] ** 2 / sizes
            right = coeffs.cardinality[idx + 1] ** 2 / (n_players - sizes)
            scores += left + right
        return _normalize_prob(np.sqrt(scores / len(semivalue_iter)))
    raise ValueError(f"Unknown coalition-size distribution: {name}")


class UtilityGame:
    """Utility oracle for bundled SOUG data and simple tabular data valuation."""

    def __init__(
        self,
        *,
        kind: str,
        n_players: int,
        sets: list[np.ndarray] | None = None,
        coefficients: np.ndarray | None = None,
        x_train=None,
        y_train=None,
        x_valid=None,
        y_valid=None,
        model: str = "logistic",
        metric: str = "accuracy",
        empty_value: float = 0.0,
        random_state: int = 0,
    ):
        self.kind = kind
        self.n_players = int(n_players)
        self.empty_value = float(empty_value)
        self.sets = sets
        self.coefficients = coefficients
        self._set_masks = None
        if kind == "soug" and sets is not None:
            self._set_masks = np.asarray(
                [sum(1 << int(player) for player in required) for required in sets],
                dtype=object,
            )
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.model = model
        self.metric = metric
        self.random_state = random_state

    @classmethod
    def from_soug_npz(cls, filename: str) -> "UtilityGame":
        data = np.load(filename)
        set_matrix = np.asarray(data["sets"], dtype=int)
        set_sizes = np.asarray(data["set_sizes"], dtype=int)
        sets = [row[:size].astype(int) for row, size in zip(set_matrix, set_sizes)]
        coefficients = np.asarray(data["coeffs"], dtype=float)
        n_players = int(data["n_player"]) if "n_player" in data else int(max(map(max, sets)) + 1)
        return cls(kind="soug", n_players=n_players, sets=sets, coefficients=coefficients)

    @classmethod
    def from_csv(
        cls,
        *,
        train_csv: str,
        valid_csv: str,
        target: str,
        model: str = "logistic",
        metric: str = "accuracy",
        random_state: int = 0,
    ) -> "UtilityGame":
        train = pd.read_csv(train_csv)
        valid = pd.read_csv(valid_csv)
        x_train = train.drop(columns=[target]).to_numpy()
        y_train = train[target].to_numpy()
        x_valid = valid.drop(columns=[target]).to_numpy()
        y_valid = valid[target].to_numpy()
        return cls(
            kind="data",
            n_players=len(train),
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            model=model,
            metric=metric,
            random_state=random_state,
        )

    def evaluate(self, coalition: np.ndarray) -> float:
        coalition = np.asarray(coalition, dtype=bool)
        if coalition.shape != (self.n_players,):
            raise ValueError(f"Coalition shape must be {(self.n_players,)}, got {coalition.shape}.")
        if not np.any(coalition):
            return self.empty_value

        if self.kind == "soug":
            mask = 0
            for player in np.flatnonzero(coalition):
                mask |= 1 << int(player)
            total = 0.0
            for required_mask, coefficient in zip(self._set_masks, self.coefficients):
                if (mask & required_mask) == required_mask:
                    total += float(coefficient)
            return total

        if self.kind == "data":
            x_train = self.x_train[coalition]
            y_train = self.y_train[coalition]
            if len(np.unique(y_train)) == 1 and self.metric == "accuracy":
                pred = np.repeat(y_train[0], len(self.y_valid))
            else:
                if self.model == "logistic":
                    estimator = LogisticRegression(max_iter=1000)
                elif self.model == "svm":
                    estimator = SVC(decision_function_shape="ovo", gamma="scale")
                elif self.model == "standardized_svm":
                    estimator = make_pipeline(
                        StandardScaler(),
                        SVC(decision_function_shape="ovo", gamma="scale"),
                    )
                elif self.model == "forest_classifier":
                    estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                elif self.model == "forest_regressor":
                    estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
                else:
                    raise ValueError(f"Unknown model: {self.model}")
                estimator.fit(x_train, y_train)
                pred = estimator.predict(self.x_valid)
            if self.metric == "accuracy":
                return float(accuracy_score(self.y_valid, pred))
            if self.metric == "negative_mse":
                return -float(mean_squared_error(self.y_valid, pred))
            raise ValueError(f"Unknown metric: {self.metric}")

        raise ValueError(f"Unknown game kind: {self.kind}")

    def soug_ground_truth(self, semivalue: str | dict) -> np.ndarray:
        if self.kind != "soug":
            raise ValueError("Closed-form SOUG ground truth is only available for SOUG games.")
        out = np.zeros(self.n_players, dtype=float)
        for required, coefficient in zip(self.sets, self.coefficients):
            factor = unanimity_semivalue_coeff(self.n_players, semivalue, len(required))
            np.add.at(out, np.asarray(required, dtype=int), float(coefficient) * factor)
        return out


@dataclass
class SharedStatistics:
    utility_sum: np.ndarray
    count: np.ndarray
    size_utility_sum: np.ndarray
    size_count: np.ndarray

    @classmethod
    def empty(cls, n_players: int) -> "SharedStatistics":
        return cls(
            utility_sum=np.zeros((n_players, n_players - 1, 2), dtype=float),
            count=np.zeros((n_players, n_players - 1, 2), dtype=float),
            size_utility_sum=np.zeros(n_players - 1, dtype=float),
            size_count=np.zeros(n_players - 1, dtype=float),
        )

    def add(self, other: "SharedStatistics") -> None:
        self.utility_sum += other.utility_sum
        self.count += other.count
        self.size_utility_sum += other.size_utility_sum
        self.size_count += other.size_count

    def copy(self) -> "SharedStatistics":
        return SharedStatistics(
            utility_sum=self.utility_sum.copy(),
            count=self.count.copy(),
            size_utility_sum=self.size_utility_sum.copy(),
            size_count=self.size_count.copy(),
        )


@dataclass
class AdaRankStatistics:
    """Shared full-sample and fold statistics used by every target semivalue."""

    full: SharedStatistics
    folds: tuple[SharedStatistics, SharedStatistics]


@dataclass
class PermutationStatistics:
    """Position-indexed marginal contributions from shared permutations."""

    marginal_sum: np.ndarray
    n_samples: int


@dataclass
class WSHAPStatistics:
    """Size-indexed marginal contributions shared across semivalues."""

    utility_sum: np.ndarray
    count: np.ndarray


def _sample_shared_statistics(
    game: UtilityGame,
    n_samples: int,
    rng: np.random.Generator,
    size_prob: np.ndarray,
    size_values: np.ndarray | None = None,
) -> SharedStatistics:
    stats = SharedStatistics.empty(game.n_players)
    size_choices = np.arange(1, game.n_players) if size_values is None else size_values
    subset_sizes = rng.choice(size_choices, size=int(n_samples), p=size_prob)
    for size in subset_sizes:
        coalition = np.zeros(game.n_players, dtype=bool)
        coalition[rng.permutation(game.n_players)[: int(size)]] = True
        value = game.evaluate(coalition)
        col = int(size) - 1
        stats.utility_sum[:, col, 0] += (~coalition) * value
        stats.utility_sum[:, col, 1] += coalition * value
        stats.count[:, col, 0] += ~coalition
        stats.count[:, col, 1] += coalition
        stats.size_utility_sum[col] += value
        stats.size_count[col] += 1
    return stats


def _add_coalition_to_stats(stats: SharedStatistics, coalition: np.ndarray, value: float) -> None:
    size = int(np.sum(coalition))
    col = size - 1
    stats.utility_sum[:, col, 0] += (~coalition) * value
    stats.utility_sum[:, col, 1] += coalition * value
    stats.count[:, col, 0] += ~coalition
    stats.count[:, col, 1] += coalition
    stats.size_utility_sum[col] += value
    stats.size_count[col] += 1


def _exact_boundary_statistics(game: UtilityGame) -> SharedStatistics:
    stats = SharedStatistics.empty(game.n_players)
    full = np.ones(game.n_players, dtype=bool)
    for player in range(game.n_players):
        singleton = np.zeros(game.n_players, dtype=bool)
        singleton[player] = True
        _add_coalition_to_stats(stats, singleton, game.evaluate(singleton))
        leave_one_out = full.copy()
        leave_one_out[player] = False
        _add_coalition_to_stats(stats, leave_one_out, game.evaluate(leave_one_out))
    return stats


def _split_samples(n_samples: int, n_blocks: int) -> list[int]:
    n_blocks = max(1, int(n_blocks))
    base = int(n_samples) // n_blocks
    rem = int(n_samples) % n_blocks
    return [base + (1 if i < rem else 0) for i in range(n_blocks)]


def _safe_average(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=float)
    mask = denominator > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + eps)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ar = rankdata(a, method="average")
    br = rankdata(b, method="average")
    return float(np.corrcoef(ar, br)[0, 1])


class SharedSamplerEstimator:
    """Base class for estimators that use shared coalition statistics."""

    cost_per_sample = 1

    def __init__(
        self,
        game: UtilityGame,
        semivalue: str | dict,
        *,
        size_distribution: str = "ofaa",
        semivalue_set: Iterable[str | dict] | None = None,
        boundary_mode: str = "none",
        empty_value: float | None = None,
        grand_value: float | None = None,
    ):
        self.game = game
        self.n_players = game.n_players
        self.semivalue = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue
        if semivalue_set is None:
            semivalue_params = [self.semivalue]
        else:
            semivalue_params = [
                semivalue_to_param(item) if isinstance(item, str) else item for item in semivalue_set
            ]
        self.coeffs = semivalue_coefficients(self.n_players, self.semivalue)
        self.boundary_mode = boundary_mode
        low, high = (2, self.n_players - 1) if boundary_mode == "exact" else (1, self.n_players)
        self.size_values = np.arange(low, high)
        self.size_prob = coalition_size_distribution(
            self.n_players,
            name=size_distribution,
            semivalues=semivalue_params,
            low=low,
            high=high,
        )
        if empty_value is None:
            empty_value = game.evaluate(np.zeros(self.n_players, dtype=bool))
        if grand_value is None:
            grand_value = game.evaluate(np.ones(self.n_players, dtype=bool))
        self.empty_value = float(empty_value)
        self.grand_value = float(grand_value)
        self.constant = self.coeffs.p[-1] * self.grand_value - self.coeffs.p[0] * self.empty_value

    def sample(self, n_samples: int, seed: int = 0) -> SharedStatistics:
        rng = np.random.default_rng(seed)
        if self.boundary_mode == "none":
            return _sample_shared_statistics(
                self.game, n_samples, rng, self.size_prob, self.size_values
            )
        if self.boundary_mode != "exact":
            raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")
        stats = _exact_boundary_statistics(self.game)
        random_samples = int(n_samples) - 2 * self.n_players
        if random_samples < 0:
            raise ValueError(
                f"Budget {n_samples} is smaller than exact boundary cost {2 * self.n_players}."
            )
        stats.add(
            _sample_shared_statistics(
                self.game, random_samples, rng, self.size_prob, self.size_values
            )
        )
        return stats

    def estimate_from_stats(self, stats: SharedStatistics) -> np.ndarray:
        raise NotImplementedError

    def run(self, n_samples: int, seed: int = 0, **kwargs) -> np.ndarray:
        return self.estimate_from_stats(self.sample(n_samples, seed=seed))

    def _include_exclude_parts(self, stats: SharedStatistics) -> tuple[np.ndarray, np.ndarray]:
        phi = _safe_average(stats.utility_sum, stats.count)
        include = np.sum(phi[:, :, 1] * self.coeffs.include_coeff, axis=1)
        exclude = np.sum(phi[:, :, 0] * self.coeffs.exclude_coeff, axis=1)
        return include, exclude

    def _size_average(self, stats: SharedStatistics) -> np.ndarray:
        return _safe_average(stats.size_utility_sum, stats.size_count)


class IncRank(SharedSamplerEstimator):
    """Include-side ranking proxy."""

    def estimate_from_stats(self, stats: SharedStatistics) -> np.ndarray:
        include, _ = self._include_exclude_parts(stats)
        size_avg = self._size_average(stats)
        offset = float(np.sum(size_avg * self.coeffs.include_constant_coeff))
        return include - offset + self.constant


class ExcRank(SharedSamplerEstimator):
    """Exclude-side ranking proxy."""

    def estimate_from_stats(self, stats: SharedStatistics) -> np.ndarray:
        _, exclude = self._include_exclude_parts(stats)
        size_avg = self._size_average(stats)
        offset = float(np.sum(size_avg * self.coeffs.exclude_constant_coeff))
        return offset - exclude + self.constant


class StdRank(SharedSamplerEstimator):
    """Standardized equal-weight combination of include and exclude signals."""

    def estimate_from_stats(self, stats: SharedStatistics) -> np.ndarray:
        include, exclude = self._include_exclude_parts(stats)
        return 0.5 * _zscore(include) - 0.5 * _zscore(exclude)


class AdaRank(SharedSamplerEstimator):
    """Two-fold split-half stability-weighted include/exclude combination."""

    def sample_adaptive_statistics(
        self,
        n_samples: int,
        seed: int = 0,
        n_blocks: int = 2,
    ) -> AdaRankStatistics:
        if int(n_blocks) != 2:
            raise ValueError("AdaRank uses the paper-facing two-fold split-half rule.")

        fold_stats: list[SharedStatistics] = []
        if self.boundary_mode == "exact":
            boundary_stats = _exact_boundary_statistics(self.game)
            full = boundary_stats.copy()
            random_samples = int(n_samples) - 2 * self.n_players
            if random_samples < 0:
                raise ValueError(
                    f"Budget {n_samples} is smaller than exact boundary cost {2 * self.n_players}."
                )
            fold_sizes = _split_samples(random_samples, 2)
        else:
            boundary_stats = None
            full = SharedStatistics.empty(self.n_players)
            fold_sizes = _split_samples(n_samples, 2)

        rng = np.random.default_rng(seed)
        for fold_n in fold_sizes:
            random_stats = _sample_shared_statistics(
                self.game,
                fold_n,
                rng,
                self.size_prob,
                self.size_values,
            )
            full.add(random_stats)
            if boundary_stats is not None:
                fold = boundary_stats.copy()
                fold.add(random_stats)
            else:
                fold = random_stats
            fold_stats.append(fold)

        return AdaRankStatistics(full=full, folds=(fold_stats[0], fold_stats[1]))

    def estimate_from_adaptive_statistics(
        self,
        stats: AdaRankStatistics,
        *,
        weight_grid: Iterable[float] | None = None,
        return_weight: bool = False,
    ):
        if weight_grid is None:
            weight_grid = np.linspace(0.0, 1.0, 21)

        full_include, full_exclude = self._include_exclude_parts(stats.full)
        full_include_z = _zscore(full_include)
        full_exclude_z = _zscore(full_exclude)
        fold0_include, fold0_exclude = self._include_exclude_parts(stats.folds[0])
        fold1_include, fold1_exclude = self._include_exclude_parts(stats.folds[1])

        best_weight = 0.5
        best_score = -np.inf
        for weight in weight_grid:
            fold0_score = weight * _zscore(fold0_include) - (1.0 - weight) * _zscore(fold0_exclude)
            fold1_score = weight * _zscore(fold1_include) - (1.0 - weight) * _zscore(fold1_exclude)
            score = _spearman(fold0_score, fold1_score)
            score -= 1e-6 * abs(float(weight) - 0.5)
            if score > best_score:
                best_score = float(score)
                best_weight = float(weight)

        estimate = best_weight * full_include_z - (1.0 - best_weight) * full_exclude_z
        if return_weight:
            return estimate, best_weight
        return estimate

    def run(
        self,
        n_samples: int,
        seed: int = 0,
        n_blocks: int = 2,
        weight_grid: Iterable[float] | None = None,
        return_weight: bool = False,
    ):
        stats = self.sample_adaptive_statistics(
            n_samples=n_samples,
            seed=seed,
            n_blocks=n_blocks,
        )
        return self.estimate_from_adaptive_statistics(
            stats,
            weight_grid=weight_grid,
            return_weight=return_weight,
        )

    def estimate_from_stats(self, stats: SharedStatistics) -> np.ndarray:
        include, exclude = self._include_exclude_parts(stats)
        return 0.5 * _zscore(include) - 0.5 * _zscore(exclude)


class OFA(SharedSamplerEstimator):
    """OFA-style shared coalition-statistic estimator."""

    def estimate_from_stats(self, stats: SharedStatistics) -> np.ndarray:
        phi = _safe_average(stats.utility_sum, stats.count)
        include = np.sum(phi[:, :, 1] * self.coeffs.ofa_include_coeff, axis=1)
        exclude = np.sum(phi[:, :, 0] * self.coeffs.ofa_exclude_coeff, axis=1)
        return self.constant + include - exclude


class OFAS(OFA):
    """Single-target OFA-S baseline."""

    def __init__(
        self,
        game: UtilityGame,
        semivalue: str | dict,
        *,
        boundary_mode: str = "none",
        **_: object,
    ):
        super().__init__(
            game,
            semivalue,
            size_distribution="ofas",
            semivalue_set=[semivalue],
            boundary_mode=boundary_mode,
        )


class GELS:
    """GELS baseline with a target-semivalue-specific sampling distribution."""

    cost_per_sample = 1

    def __init__(self, game: UtilityGame, semivalue: str | dict):
        self.game = game
        self.n_players = game.n_players
        param = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue
        self.p = semivalue_probabilities(self.n_players, param)[0]
        sizes = np.arange(1, self.n_players + 1, dtype=float)
        log_p = np.full(self.n_players, -np.inf, dtype=float)
        positive = self.p > 0
        log_p[positive] = np.log(self.p[positive])
        log_q = _log_comb(self.n_players + 1, sizes) + log_p
        self.size_prob = np.exp(log_q - special.logsumexp(log_q))
        weighted_sum = special.logsumexp(log_q + np.log(sizes))
        self.gels_constant = float(np.exp(weighted_sum) / (self.n_players + 1))

    def run(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        utility_sum = np.zeros(self.n_players + 1, dtype=float)
        count = np.zeros(self.n_players + 1, dtype=float)
        sizes = np.arange(1, self.n_players + 1)
        for _ in range(int(n_samples)):
            size = int(rng.choice(sizes, p=self.size_prob))
            selected = rng.choice(self.n_players + 1, size=size, replace=False)
            coalition = np.zeros(self.n_players, dtype=bool)
            original_players = selected[selected < self.n_players]
            coalition[original_players] = True
            value = self.game.evaluate(coalition)
            utility_sum[selected] += value
            count[selected] += 1.0
        average = _safe_average(utility_sum, count)
        return (average[: self.n_players] - average[self.n_players]) * self.gels_constant


class ExactValue:
    """Exact semivalue computation for small games."""

    def __init__(self, game: UtilityGame, semivalue: str | dict):
        self.game = game
        self.n_players = game.n_players
        self.semivalue = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue
        self.coeffs = semivalue_coefficients(self.n_players, self.semivalue)

    def run(self) -> np.ndarray:
        values = np.zeros(self.n_players, dtype=float)
        base = np.zeros(self.n_players, dtype=bool)
        with_player = np.zeros(self.n_players, dtype=bool)
        for subset in product([False, True], repeat=self.n_players - 1):
            size_without = sum(subset)
            weight = self.coeffs.p[size_without]
            base[: self.n_players - 1] = subset
            with_player[: self.n_players - 1] = subset
            with_player[-1] = True
            base[-1] = False
            values[-1] += weight * (self.game.evaluate(with_player) - self.game.evaluate(base))
            for player in range(self.n_players - 1):
                base[player], base[-1] = base[-1], base[player]
                with_player[player], with_player[-1] = with_player[-1], with_player[player]
                values[player] += weight * (self.game.evaluate(with_player) - self.game.evaluate(base))
                base[player], base[-1] = base[-1], base[player]
                with_player[player], with_player[-1] = with_player[-1], with_player[player]
        return values


class WPERM:
    """Weighted permutation baseline."""

    cost_per_sample = "n"

    def __init__(self, game: UtilityGame, semivalue: str | dict):
        self.game = game
        self.n_players = game.n_players
        param = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue
        self.weights = semivalue_coefficients(self.n_players, param).cardinality * self.n_players

    def sample(self, n_samples: int, seed: int = 0) -> PermutationStatistics:
        rng = np.random.default_rng(seed)
        marginal_sum = np.zeros((self.n_players, self.n_players), dtype=float)
        empty = np.zeros(self.n_players, dtype=bool)
        empty_value = self.game.evaluate(empty)
        for _ in range(int(n_samples)):
            coalition = np.zeros(self.n_players, dtype=bool)
            old_value = empty_value
            for position, player in enumerate(rng.permutation(self.n_players)):
                coalition[player] = True
                new_value = self.game.evaluate(coalition)
                marginal_sum[player, position] += new_value - old_value
                old_value = new_value
        return PermutationStatistics(marginal_sum=marginal_sum, n_samples=int(n_samples))

    def estimate_from_stats(self, stats: PermutationStatistics) -> np.ndarray:
        weighted = np.sum(stats.marginal_sum * self.weights[None, :], axis=1)
        return weighted / max(1, stats.n_samples)

    def run(self, n_samples: int, seed: int = 0) -> np.ndarray:
        return self.estimate_from_stats(self.sample(n_samples=n_samples, seed=seed))


class WSL:
    """Weighted sampling and learning-style baseline."""

    cost_per_sample = "2n"

    def __init__(self, game: UtilityGame, semivalue: str | dict):
        self.game = game
        self.n_players = game.n_players
        self.semivalue = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue

    def _draw_without_target(self, rng: np.random.Generator) -> np.ndarray:
        kind = self.semivalue["kind"]
        if kind == "BZ":
            q = 0.5
        elif kind == "WBZ":
            q = float(self.semivalue["q"])
        elif kind == "SV":
            q = float(rng.random())
        elif kind == "BT":
            q = float(rng.beta(self.semivalue["beta"], self.semivalue["alpha"]))
        else:
            raise ValueError(f"Unsupported semivalue kind: {kind}")
        return rng.binomial(1, q, size=self.n_players - 1).astype(bool)

    def run(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = np.zeros(self.n_players, dtype=float)
        coalition = np.zeros(self.n_players, dtype=bool)
        for _ in range(int(n_samples)):
            sample = self._draw_without_target(rng)
            coalition[: self.n_players - 1] = sample
            coalition[-1] = False
            out[-1] -= self.game.evaluate(coalition)
            coalition[-1] = True
            out[-1] += self.game.evaluate(coalition)
            for player in range(self.n_players - 1):
                coalition[player], coalition[-1] = coalition[-1], coalition[player]
                out[player] += self.game.evaluate(coalition)
                coalition[player] = False
                out[player] -= self.game.evaluate(coalition)
                coalition[player] = True
                coalition[player], coalition[-1] = coalition[-1], coalition[player]
            coalition[-1] = False
        return out / max(1, int(n_samples))


class WSHAP:
    """Weighted Shapley-style local marginal baseline."""

    cost_per_sample = "2n(n-1)"

    def __init__(self, game: UtilityGame, semivalue: str | dict):
        self.game = game
        self.n_players = game.n_players
        param = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue
        self.weights = semivalue_coefficients(self.n_players, param).cardinality

    def sample(self, n_samples: int, seed: int = 0) -> WSHAPStatistics:
        rng = np.random.default_rng(seed)
        utility_sum = np.zeros((self.n_players, self.n_players), dtype=float)
        count = np.zeros((self.n_players, self.n_players), dtype=float)
        for _ in range(int(n_samples)):
            for player in range(self.n_players):
                coalition = np.zeros(self.n_players, dtype=bool)
                for candidate in rng.permutation(self.n_players):
                    if candidate == player:
                        continue
                    coalition[candidate] = True
                    size = int(np.sum(coalition))
                    without = self.game.evaluate(coalition)
                    coalition[player] = True
                    with_player = self.game.evaluate(coalition)
                    coalition[player] = False
                    utility_sum[player, size] += with_player - without
                    count[player, size] += 1
        return WSHAPStatistics(utility_sum=utility_sum, count=count)

    def estimate_from_stats(self, stats: WSHAPStatistics) -> np.ndarray:
        return np.sum(_safe_average(stats.utility_sum, stats.count) * self.weights, axis=1)

    def run(self, n_samples: int, seed: int = 0) -> np.ndarray:
        return self.estimate_from_stats(self.sample(n_samples=n_samples, seed=seed))


class SHAPIQ:
    """Coalition-statistic SHAP-IQ-style baseline."""

    cost_per_sample = 1

    def __init__(
        self,
        game: UtilityGame,
        semivalue: str | dict,
        *,
        empty_value: float | None = None,
        grand_value: float | None = None,
    ):
        self.game = game
        self.n_players = game.n_players
        param = semivalue_to_param(semivalue) if isinstance(semivalue, str) else semivalue
        coeffs = semivalue_coefficients(self.n_players, param)
        sizes = np.arange(1, self.n_players, dtype=np.float64)
        self.size_prob = _normalize_prob(1.0 / (sizes * (self.n_players - sizes)))
        self.weights_positive = coeffs.cardinality * self.n_players / np.arange(1, self.n_players + 1)
        self.weights_negative = coeffs.cardinality * self.n_players / np.arange(self.n_players, 0, -1)
        if empty_value is None:
            empty_value = game.evaluate(np.zeros(self.n_players, dtype=bool))
        if grand_value is None:
            grand_value = game.evaluate(np.ones(self.n_players, dtype=bool))
        self.empty_value = float(empty_value)
        self.grand_value = float(grand_value)
        self.constant = coeffs.p[-1] * self.grand_value - coeffs.p[0] * self.empty_value

    def sample(self, n_samples: int, seed: int = 0) -> SharedStatistics:
        rng = np.random.default_rng(seed)
        stats = SharedStatistics.empty(self.n_players)
        sizes = np.arange(1, self.n_players)
        for _ in range(int(n_samples)):
            size = int(rng.choice(sizes, p=self.size_prob))
            coalition = np.zeros(self.n_players, dtype=bool)
            coalition[rng.choice(self.n_players, size=size, replace=False)] = True
            value = self.game.evaluate(coalition)
            _add_coalition_to_stats(stats, coalition, value)
        return stats

    def estimate_from_stats(self, stats: SharedStatistics) -> np.ndarray:
        positive_scale = self.weights_positive[:-1] / self.size_prob
        negative_scale = self.weights_negative[1:] / self.size_prob
        out = np.sum(stats.utility_sum[:, :, 1] * positive_scale[None, :], axis=1)
        out -= np.sum(stats.utility_sum[:, :, 0] * negative_scale[None, :], axis=1)
        n_samples = int(np.sum(stats.size_count))
        return out / max(1, n_samples) + self.constant

    def run(self, n_samples: int, seed: int = 0) -> np.ndarray:
        return self.estimate_from_stats(self.sample(n_samples=n_samples, seed=seed))


PAPER_METHODS = {
    "IncRank": IncRank,
    "ExcRank": ExcRank,
    "StdRank": StdRank,
    "AdaRank": AdaRank,
    "OFA": OFA,
    "OFA-S": OFAS,
    "GELS": GELS,
    "WPERM": WPERM,
    "WSL": WSL,
    "WSHAP": WSHAP,
    "SHAPIQ": SHAPIQ,
}


def samples_from_budget(method_name: str, n_players: int, utility_budget: int) -> int:
    """Convert a utility-evaluation budget into sampled objects for each method."""
    if method_name == "WPERM":
        return max(1, int(utility_budget) // n_players)
    if method_name == "WSL":
        return max(1, int(utility_budget) // (2 * n_players))
    if method_name == "WSHAP":
        return max(1, int(utility_budget) // (2 * n_players * max(1, n_players - 1)))
    return max(1, int(utility_budget))


def budget_from_uep(method_name: str, n_players: int, utility_evaluations_per_player: int) -> int:
    return int(n_players) * int(utility_evaluations_per_player)


def minimum_uep_for_one_sample(method_name: str, n_players: int) -> int:
    if method_name == "WPERM":
        return 1
    if method_name == "WSL":
        return 2
    if method_name == "WSHAP":
        return int(ceil(2 * max(1, n_players - 1)))
    return 1
