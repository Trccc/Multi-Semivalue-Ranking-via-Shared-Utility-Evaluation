from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source import SEMIVALUE_SET, UtilityGame, semivalue_coefficients, semivalue_to_param


DATASETS = {
    "breast_cancer": {
        "label": "Breast Cancer",
        "directory": "breast_cancer_n50",
        "model": "svm",
        "seed_parts": ("breast_cancer_wperm_reference", "dataval_large", 50),
    },
    "wine": {
        "label": "Wine",
        "directory": "wine_n50",
        "model": "standardized_svm",
        "seed_parts": ("wine_wperm_reference", 50),
    },
}


def stable_seed(*parts: object) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def build_game(dataset: str) -> UtilityGame:
    setting = DATASETS[dataset]
    data_dir = ROOT / "data" / setting["directory"]
    return UtilityGame.from_csv(
        train_csv=data_dir / "players.csv",
        valid_csv=data_dir / "validation.csv",
        target="Y",
        model=setting["model"],
        metric="accuracy",
    )


class CachedMaskUtility:
    def __init__(self, dataset: str) -> None:
        self.game = build_game(dataset)
        self.n = self.game.n_players
        self.cache: dict[int, float] = {}

    def __call__(self, mask: int) -> float:
        cached = self.cache.get(mask)
        if cached is not None:
            return cached
        coalition = np.fromiter(
            ((mask >> player) & 1 for player in range(self.n)),
            dtype=bool,
            count=self.n,
        )
        value = float(self.game.evaluate(coalition))
        self.cache[mask] = value
        return value


def score_from_stats(
    n: int,
    semivalue: str,
    contribution_by_position: np.ndarray,
    permutations: int,
) -> np.ndarray:
    coefficients = semivalue_coefficients(n, semivalue_to_param(semivalue))
    return contribution_by_position @ (coefficients.cardinality * n) / float(permutations)


def worker_contributions(
    dataset: str,
    batch_seed: int,
    start_permutation: int,
    stop_permutation: int,
) -> np.ndarray:
    utility = CachedMaskUtility(dataset)
    n = utility.n
    contribution_by_position = np.zeros((n, n), dtype=float)
    empty_value = utility(0)
    for permutation_index in range(start_permutation, stop_permutation):
        rng = np.random.default_rng(stable_seed(batch_seed, permutation_index))
        permutation = rng.permutation(n)
        mask = 0
        previous = empty_value
        for position, player in enumerate(permutation):
            mask |= 1 << int(player)
            value = utility(mask)
            contribution_by_position[int(player), position] += value - previous
            previous = value
    return contribution_by_position


def worker_from_args(args: tuple[str, int, int, int]) -> np.ndarray:
    return worker_contributions(*args)


def split_ranges(start: int, stop: int, workers: int) -> list[tuple[int, int]]:
    count = stop - start
    active_workers = min(max(1, workers), count)
    base, remainder = divmod(count, active_workers)
    ranges = []
    current = start
    for worker in range(active_workers):
        next_current = current + base + int(worker < remainder)
        ranges.append((current, next_current))
        current = next_current
    return ranges


def batch_stem(dataset: str, batch_index: int, batch_seed: int) -> str:
    return f"{dataset}_batch{batch_index}_seed{batch_seed}"


def latest_batch_file(
    batch_dir: Path,
    dataset: str,
    batch_index: int,
    batch_seed: int,
    target_permutations: int,
) -> Path | None:
    prefix = batch_stem(dataset, batch_index, batch_seed) + "_p"
    candidates = []
    for candidate in batch_dir.glob(f"{prefix}*.npz"):
        suffix = candidate.stem.removeprefix(prefix)
        if suffix.isdigit() and int(suffix) <= target_permutations:
            candidates.append((int(suffix), candidate))
    return max(candidates, default=(None, None))[1]


def load_or_extend_batch(
    dataset: str,
    batch_dir: Path,
    batch_index: int,
    batch_seed: int,
    target_permutations: int,
    workers: int,
) -> tuple[np.ndarray, float, Path]:
    previous_path = latest_batch_file(
        batch_dir,
        dataset,
        batch_index,
        batch_seed,
        target_permutations,
    )
    if previous_path is None:
        start_permutation = 0
        contribution_by_position = np.zeros((50, 50), dtype=float)
    else:
        with np.load(previous_path, allow_pickle=False) as loaded:
            start_permutation = int(loaded["n_permutations"])
            contribution_by_position = np.asarray(
                loaded["contribution_by_position"], dtype=float
            )
    output_path = batch_dir / (
        f"{batch_stem(dataset, batch_index, batch_seed)}_p{target_permutations}.npz"
    )
    if start_permutation == target_permutations:
        return contribution_by_position, 0.0, output_path
    started = perf_counter()
    ranges = split_ranges(start_permutation, target_permutations, workers)
    tasks = [
        (dataset, batch_seed, begin, end)
        for begin, end in ranges
    ]
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        for partial in executor.map(worker_from_args, tasks):
            contribution_by_position += partial
    elapsed = perf_counter() - started
    np.savez_compressed(
        output_path,
        contribution_by_position=contribution_by_position,
        n_permutations=target_permutations,
        batch_index=batch_index,
        batch_seed=batch_seed,
        n_player=50,
        extended_from_permutations=start_permutation,
        elapsed_seconds=elapsed,
    )
    return contribution_by_position, elapsed, output_path


def rank_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(left, right).statistic),
        "kendall": float(kendalltau(left, right).statistic),
        "max_abs": float(np.max(np.abs(left - right))),
        "mean_abs": float(np.mean(np.abs(left - right))),
    }


def quality_tables(
    batch_scores: np.ndarray,
    permutations: int,
    min_spearman: float,
    min_kendall: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_rows = []
    summary_rows = []
    for semivalue_index, semivalue in enumerate(SEMIVALUE_SET):
        metrics = []
        for left, right in combinations(range(batch_scores.shape[0]), 2):
            row_metrics = rank_metrics(
                batch_scores[left, semivalue_index],
                batch_scores[right, semivalue_index],
            )
            metrics.append(row_metrics)
            pairwise_rows.append(
                {
                    "n": 50,
                    "semivalue": semivalue,
                    "permutations_per_raw_batch": permutations,
                    "raw_batches_per_group": 1,
                    "group_a": left,
                    "group_b": right,
                    **row_metrics,
                }
            )
        minimum_rho = min(item["spearman"] for item in metrics)
        minimum_tau = min(item["kendall"] for item in metrics)
        summary_rows.append(
            {
                "n": 50,
                "semivalue": semivalue,
                "permutations_per_raw_batch": permutations,
                "raw_batches": batch_scores.shape[0],
                "group_size": 1,
                "n_independent_groups": batch_scores.shape[0],
                "min_spearman": minimum_rho,
                "min_kendall": minimum_tau,
                "mean_spearman": float(
                    np.mean([item["spearman"] for item in metrics])
                ),
                "mean_kendall": float(
                    np.mean([item["kendall"] for item in metrics])
                ),
                "quality_decision": (
                    "PASS"
                    if minimum_rho >= min_spearman and minimum_tau >= min_kendall
                    else "FAIL"
                ),
            }
        )
    return pd.DataFrame(pairwise_rows), pd.DataFrame(summary_rows)


def resolve_settings(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.preset == "paper":
        permutations = 100000
        batches = 3
        workers = min(22, os.cpu_count() or 1)
    else:
        permutations = 2
        batches = 2
        workers = min(2, os.cpu_count() or 1)
    return (
        permutations if args.permutations is None else args.permutations,
        batches if args.batches is None else args.batches,
        workers if args.workers is None else args.workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an n=50 WPERM reference.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--preset", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--permutations", type=int)
    parser.add_argument("--batches", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--min-spearman", type=float, default=0.90)
    parser.add_argument("--min-kendall", type=float, default=0.75)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "references")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    permutations, batches, workers = resolve_settings(args)
    if permutations < 1 or batches < 2 or workers < 1:
        raise ValueError("Positive permutations/workers and at least two batches are required.")
    setting = DATASETS[args.dataset]
    run_dir = args.output_root / args.dataset / f"p{permutations}_b{batches}"
    batch_dir = args.output_root / args.dataset / "batches"
    reference_path = run_dir / "reference_scores.npz"
    if reference_path.exists() and not args.overwrite:
        print(f"exists: {reference_path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = []
    batch_records = []
    for batch_index in range(batches):
        batch_seed = stable_seed(args.seed, *setting["seed_parts"], batch_index)
        print(
            f"batch {batch_index + 1}/{batches}: seed={batch_seed}, "
            f"permutations={permutations}",
            flush=True,
        )
        statistics, elapsed, batch_path = load_or_extend_batch(
            args.dataset,
            batch_dir,
            batch_index,
            batch_seed,
            permutations,
            workers,
        )
        raw_scores.append(
            np.vstack(
                [
                    score_from_stats(50, semivalue, statistics, permutations)
                    for semivalue in SEMIVALUE_SET
                ]
            )
        )
        batch_records.append(
            {
                "batch_index": batch_index,
                "seed": batch_seed,
                "permutations": permutations,
                "utility_evaluations": 50 * permutations,
                "elapsed_seconds": elapsed,
                "statistics_file": str(batch_path),
            }
        )

    batch_scores = np.asarray(raw_scores)
    scores = batch_scores.mean(axis=0)
    rankings = np.vstack([rankdata(row, method="average") for row in scores])
    pairwise, summary = quality_tables(
        batch_scores,
        permutations,
        args.min_spearman,
        args.min_kendall,
    )
    decision = "PASS" if summary["quality_decision"].eq("PASS").all() else "FAIL"
    np.savez_compressed(
        reference_path,
        semivalue_ids=np.asarray(SEMIVALUE_SET),
        scores=scores,
        rankings=rankings,
        batch_scores=batch_scores,
        permutations_per_raw_batch=permutations,
        n_raw_batches=batches,
        group_size=1,
        n_player=50,
        sampler="WPERM uniform permutations",
    )
    pairwise.to_csv(run_dir / "quality_pairwise.csv", index=False)
    summary.to_csv(run_dir / "quality_summary.csv", index=False)
    manifest = {
        "dataset": setting["label"],
        "model": setting["model"],
        "n_player": 50,
        "semivalue_ids": list(SEMIVALUE_SET),
        "sampler": "WPERM uniform permutations",
        "seed": args.seed,
        "permutations_per_raw_batch": permutations,
        "n_raw_batches": batches,
        "utility_evaluations_per_raw_batch": 50 * permutations,
        "quality_gate": {
            "min_spearman": args.min_spearman,
            "min_kendall": args.min_kendall,
        },
        "quality_decision": decision,
        "batch_records": batch_records,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"wrote: {reference_path}")
    print(f"quality: {decision}")


if __name__ == "__main__":
    main()
