from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper import pairwise_accuracy, spearman_rho, topk_overlap
from source import (
    PAPER_METHODS,
    SEMIVALUE_SET,
    UtilityGame,
    minimum_uep_for_one_sample,
    samples_from_budget,
)


SHARED_STATISTIC_METHODS = {"IncRank", "ExcRank", "StdRank", "OFA"}
SHARED_METHODS = SHARED_STATISTIC_METHODS | {
    "AdaRank",
    "WPERM",
    "WSHAP",
    "SHAPIQ",
}
TARGET_SPECIFIC_METHODS = {"OFA-S", "GELS", "WSL"}
EXACT_BOUNDARY_METHODS = SHARED_STATISTIC_METHODS | {"AdaRank", "OFA-S"}


def build_game(dataset: str) -> UtilityGame:
    if dataset == "soug":
        return UtilityGame.from_soug_npz(ROOT / "data" / "soug_n50" / "soug_n50.npz")
    if dataset == "breast_cancer":
        return UtilityGame.from_csv(
            train_csv=ROOT / "data" / "breast_cancer_n50" / "players.csv",
            valid_csv=ROOT / "data" / "breast_cancer_n50" / "validation.csv",
            target="Y",
            model="svm",
            metric="accuracy",
        )
    if dataset == "wine":
        return UtilityGame.from_csv(
            train_csv=ROOT / "data" / "wine_n50" / "players.csv",
            valid_csv=ROOT / "data" / "wine_n50" / "validation.csv",
            target="Y",
            model="standardized_svm",
            metric="accuracy",
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def load_references(
    dataset: str,
    game: UtilityGame,
    semivalues: list[str],
) -> dict[str, np.ndarray]:
    if dataset == "soug":
        return {
            semivalue: game.soug_ground_truth(semivalue)
            for semivalue in semivalues
        }
    reference_file = ROOT / "data" / f"{dataset}_n50" / "reference_scores.npz"
    with np.load(reference_file, allow_pickle=False) as loaded:
        reference_ids = [str(item) for item in loaded["semivalue_ids"]]
        scores = np.asarray(loaded["scores"], dtype=float)
    missing = sorted(set(semivalues) - set(reference_ids))
    if missing:
        raise ValueError(f"Reference does not contain: {missing}")
    return {
        semivalue: scores[reference_ids.index(semivalue)]
        for semivalue in semivalues
    }


def shared_estimates(
    game: UtilityGame,
    method: str,
    semivalues: list[str],
    n_samples: int,
    seed: int,
    size_distribution: str,
    size_distribution_alpha: float,
    boundary_mode: str,
) -> dict[str, np.ndarray]:
    method_class = PAPER_METHODS[method]
    if method in SHARED_STATISTIC_METHODS | {"AdaRank"}:
        common = {
            "size_distribution": size_distribution,
            "size_distribution_alpha": size_distribution_alpha,
            "semivalue_set": semivalues,
            "boundary_mode": boundary_mode,
        }
        first = method_class(game, semivalues[0], **common)
        estimators = [first]
        for semivalue in semivalues[1:]:
            estimators.append(
                method_class(
                    game,
                    semivalue,
                    empty_value=first.empty_value,
                    grand_value=first.grand_value,
                    **common,
                )
            )
        if method == "AdaRank":
            statistics = first.sample_adaptive_statistics(n_samples=n_samples, seed=seed)
            return {
                semivalue: estimator.estimate_from_adaptive_statistics(statistics)
                for semivalue, estimator in zip(semivalues, estimators)
            }
        statistics = first.sample(n_samples=n_samples, seed=seed)
        return {
            semivalue: estimator.estimate_from_stats(statistics)
            for semivalue, estimator in zip(semivalues, estimators)
        }

    first = method_class(game, semivalues[0])
    estimators = [first]
    for semivalue in semivalues[1:]:
        if method == "SHAPIQ":
            estimators.append(
                method_class(
                    game,
                    semivalue,
                    empty_value=first.empty_value,
                    grand_value=first.grand_value,
                )
            )
        else:
            estimators.append(method_class(game, semivalue))
    statistics = first.sample(n_samples=n_samples, seed=seed)
    return {
        semivalue: estimator.estimate_from_stats(statistics)
        for semivalue, estimator in zip(semivalues, estimators)
    }


def target_specific_estimate(
    game: UtilityGame,
    method: str,
    semivalue: str,
    n_samples: int,
    seed: int,
    boundary_mode: str,
) -> np.ndarray:
    kwargs = {"boundary_mode": boundary_mode} if method == "OFA-S" else {}
    return PAPER_METHODS[method](game, semivalue, **kwargs).run(
        n_samples=n_samples,
        seed=seed,
    )


def validate_config(
    config: dict,
) -> tuple[str, str, list[str], int, int, str, float, str]:
    dataset = str(config["dataset"])
    method = str(config["method"])
    semivalues = [str(item) for item in config["semivalues"]]
    uep = int(config["uep"])
    seed = int(config["seed"])
    size_distribution = str(config.get("size_distribution", "ofaa"))
    size_distribution_alpha = float(config.get("size_distribution_alpha", 0.0))
    boundary_mode = str(config.get("boundary_mode", "exact"))
    if dataset not in {"soug", "breast_cancer", "wine"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    if method not in PAPER_METHODS:
        raise ValueError(f"Unknown method: {method}")
    if not semivalues:
        raise ValueError("At least one semivalue is required.")
    unknown_semivalues = sorted(set(semivalues) - set(SEMIVALUE_SET))
    if unknown_semivalues:
        raise ValueError(f"Unknown semivalues: {unknown_semivalues}")
    if len(set(semivalues)) != len(semivalues):
        raise ValueError("Semivalues must be unique.")
    if method in TARGET_SPECIFIC_METHODS and len(semivalues) != 1:
        raise ValueError(f"{method} requires exactly one semivalue.")
    if uep < 1:
        raise ValueError("uep must be positive.")
    if not np.isfinite(size_distribution_alpha):
        raise ValueError("size_distribution_alpha must be finite.")
    if boundary_mode not in {"exact", "none"}:
        raise ValueError(f"Unknown boundary mode: {boundary_mode}")
    return (
        dataset,
        method,
        semivalues,
        uep,
        seed,
        size_distribution,
        size_distribution_alpha,
        boundary_mode,
    )


def run_sampling(config: dict) -> dict:
    (
        dataset,
        method,
        semivalues,
        uep,
        seed,
        size_distribution,
        size_distribution_alpha,
        boundary_mode,
    ) = validate_config(config)
    game = build_game(dataset)
    references = load_references(dataset, game, semivalues)
    total_budget = game.n_players * uep
    minimum_budget = game.n_players * minimum_uep_for_one_sample(
        method,
        game.n_players,
    )
    if boundary_mode == "exact" and method in EXACT_BOUNDARY_METHODS:
        minimum_budget = max(minimum_budget, 2 * game.n_players)
    if total_budget < minimum_budget:
        return {
            "dataset": dataset,
            "method": method,
            "semivalues": semivalues,
            "uep": uep,
            "seed": seed,
            "size_distribution": size_distribution,
            "size_distribution_alpha": size_distribution_alpha,
            "boundary_mode": boundary_mode,
            "total_budget": total_budget,
            "n_samples": 0,
            "sampling_passes": 0,
            "runtime_seconds": 0.0,
            "status": "not_applicable",
            "estimates": {},
            "references": references,
            "metrics": pd.DataFrame(),
        }

    n_samples = samples_from_budget(method, game.n_players, total_budget)
    started = perf_counter()
    if method in SHARED_METHODS:
        estimates = shared_estimates(
            game,
            method,
            semivalues,
            n_samples,
            seed,
            size_distribution,
            size_distribution_alpha,
            boundary_mode,
        )
    else:
        semivalue = semivalues[0]
        estimates = {
            semivalue: target_specific_estimate(
                game,
                method,
                semivalue,
                n_samples,
                seed,
                boundary_mode,
            )
        }
    runtime = perf_counter() - started
    rows = []
    for semivalue in semivalues:
        reference = references[semivalue]
        estimate = estimates[semivalue]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "semivalue": semivalue,
                "uep": uep,
                "seed": seed,
                "total_budget": total_budget,
                "n_samples": n_samples,
                "rho": spearman_rho(reference, estimate),
                "pairwise": pairwise_accuracy(reference, estimate),
                "topk": topk_overlap(reference, estimate),
                "status": "ok",
            }
        )
    return {
        "dataset": dataset,
        "method": method,
        "semivalues": semivalues,
        "uep": uep,
        "seed": seed,
        "size_distribution": size_distribution,
        "size_distribution_alpha": size_distribution_alpha,
        "boundary_mode": boundary_mode,
        "total_budget": total_budget,
        "n_samples": n_samples,
        "sampling_passes": 1,
        "runtime_seconds": runtime,
        "status": "ok",
        "estimates": estimates,
        "references": references,
        "metrics": pd.DataFrame(rows),
    }


def save_result(result: dict, output_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_name}_{stamp}"
    counter = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_name}_{stamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True)
    result["metrics"].to_csv(run_dir / "metrics.csv", index=False)
    if result["status"] == "ok":
        semivalues = result["semivalues"]
        np.savez_compressed(
            run_dir / "scores.npz",
            semivalue_ids=np.asarray(semivalues),
            estimates=np.vstack([result["estimates"][item] for item in semivalues]),
            references=np.vstack([result["references"][item] for item in semivalues]),
        )
    manifest = {
        key: result[key]
        for key in [
            "dataset",
            "method",
            "semivalues",
            "uep",
            "seed",
            "size_distribution",
            "size_distribution_alpha",
            "boundary_mode",
            "total_budget",
            "n_samples",
            "sampling_passes",
            "runtime_seconds",
            "status",
        ]
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one sampling pass from a JSON config.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "sampling.json")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    result = run_sampling(config)
    if result["metrics"].empty:
        print(result["status"])
    else:
        print(result["metrics"].to_string(index=False))
    print(f"sampling_passes: {result['sampling_passes']}")
    if not args.no_save:
        output_root = ROOT / str(config.get("output_root", "outputs/sampling"))
        run_name = str(config.get("run_name", "sampling"))
        print(f"wrote: {save_result(result, output_root, run_name)}")


if __name__ == "__main__":
    main()
