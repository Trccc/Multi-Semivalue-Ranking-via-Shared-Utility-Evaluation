from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multi_semi_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper import log_aurc
from scripts.run_sampling import run_sampling
from source import SEMIVALUE_SET


METHODS = ["AdaRank", "OFA", "WPERM", "SHAPIQ"]
METHOD_LABELS = {
    "AdaRank": "AdaRank",
    "OFA": "OFA",
    "WPERM": "WPERM",
    "SHAPIQ": "SHAP-IQ",
}
METHOD_COLORS = {
    "AdaRank": "#4C78A8",
    "OFA": "#E45756",
    "WPERM": "#54A24B",
    "SHAPIQ": "#B279A2",
}
DATASET_LABELS = {
    "soug": "SOUG",
    "breast_cancer": "Breast Cancer",
    "wine": "Wine",
}
FAMILIES = [
    ("BT", ["BT-1-1", "BT-1-4", "BT-4-1", "BT-4-4"]),
    ("WBZ", ["WBZ-01", "WBZ-03", "WBZ-05", "WBZ-07", "WBZ-09"]),
    ("CAN", ["BT-1-1", "WBZ-05"]),
    ("ALL", list(SEMIVALUE_SET)),
]


def stable_seed(*parts: object) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def split_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_matrix(dataset: str, budgets: list[int], runs: int) -> pd.DataFrame:
    rows = []
    total = len(METHODS) * len(budgets) * runs
    completed = 0
    for method in METHODS:
        for uep in budgets:
            for run in range(runs):
                seed = stable_seed(20260702, "figure", dataset, method, uep, run)
                result = run_sampling(
                    {
                        "dataset": dataset,
                        "method": method,
                        "semivalues": list(SEMIVALUE_SET),
                        "uep": uep,
                        "seed": seed,
                        "size_distribution": "ofaa",
                        "boundary_mode": "exact",
                    }
                )
                metrics = result["metrics"].copy()
                metrics["run"] = run
                metrics["runtime_seconds"] = result["runtime_seconds"]
                rows.append(metrics)
                completed += 1
                print(
                    f"completed {completed}/{total}: {dataset} {method} uep={uep} run={run}",
                    flush=True,
                )
    return pd.concat(rows, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, semivalues in FAMILIES:
        family_results = results[results["semivalue"].isin(semivalues)]
        central_rows = []
        unit_rows = []
        for (method, semivalue), group in family_results.groupby(
            ["method", "semivalue"], sort=False
        ):
            curve = group.groupby("uep", as_index=False).agg(rho=("rho", "mean"))
            central_rows.append(
                {
                    "method": method,
                    "semivalue": semivalue,
                    "log_aurc": log_aurc(curve),
                }
            )
        for (method, run, semivalue), group in family_results.groupby(
            ["method", "run", "semivalue"], sort=False
        ):
            unit_rows.append(
                {
                    "method": method,
                    "run": int(run),
                    "semivalue": semivalue,
                    "log_aurc": log_aurc(group[["uep", "rho"]]),
                }
            )
        central = pd.DataFrame(central_rows)
        units = pd.DataFrame(unit_rows)
        unit_families = (
            units.groupby(["method", "run"], as_index=False)
            .agg(avg_log_aurc=("log_aurc", "mean"))
        )
        for method in METHODS:
            method_central = central[central["method"].eq(method)]
            method_units = unit_families[unit_families["method"].eq(method)]
            n_units = len(method_units)
            standard_deviation = (
                float(method_units["avg_log_aurc"].std(ddof=1))
                if n_units > 1
                else 0.0
            )
            rows.append(
                {
                    "family": family,
                    "method": method,
                    "avg_log_aurc": float(method_central["log_aurc"].mean()),
                    "avg_log_aurc_se": (
                        standard_deviation / np.sqrt(n_units) if n_units else np.nan
                    ),
                    "worst_log_aurc": float(method_central["log_aurc"].min()),
                    "n_runs": n_units,
                    "n_semivalues": len(method_central),
                }
            )
    summary = pd.DataFrame(rows)
    family_order = {name: index for index, (name, _) in enumerate(FAMILIES)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    summary["_family"] = summary["family"].map(family_order)
    summary["_method"] = summary["method"].map(method_order)
    return (
        summary.sort_values(["_family", "_method"])
        .drop(columns=["_family", "_method"])
        .reset_index(drop=True)
    )


def metric_limits(summary: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate(
        [
            summary["avg_log_aurc"].to_numpy(dtype=float),
            summary["worst_log_aurc"].to_numpy(dtype=float),
            (
                summary["avg_log_aurc"] + summary["avg_log_aurc_se"].fillna(0.0)
            ).to_numpy(dtype=float),
        ]
    )
    finite = values[np.isfinite(values)]
    lower = min(0.0, float(finite.min()) - 0.04)
    upper = float(finite.max()) + 0.04
    lower = max(-0.25, np.floor(lower / 0.05) * 0.05)
    upper = min(1.04, np.ceil(upper / 0.05) * 0.05)
    if upper - lower < 0.2:
        upper = min(1.04, lower + 0.2)
    return float(lower), float(upper)


def plot(summary: pd.DataFrame, dataset: str, output: Path) -> None:
    family_names = [name for name, _ in FAMILIES]
    x = np.arange(len(family_names), dtype=float)
    width = 0.17
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for method_index, method in enumerate(METHODS):
        method_data = (
            summary[summary["method"].eq(method)]
            .set_index("family")
            .reindex(family_names)
        )
        positions = x + offsets[method_index]
        averages = method_data["avg_log_aurc"].to_numpy(dtype=float)
        errors = method_data["avg_log_aurc_se"].fillna(0.0).to_numpy(dtype=float)
        worst = method_data["worst_log_aurc"].to_numpy(dtype=float)
        ax.bar(
            positions,
            averages,
            width=width * 0.92,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            yerr=errors,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 0.65,
                "capsize": 1.4,
                "capthick": 0.65,
            },
        )
        for position, value in zip(positions, worst):
            ax.hlines(
                value,
                position - width * 0.33,
                position + width * 0.33,
                color="#111111",
                linewidth=1.25,
                zorder=5,
            )

    fig.suptitle(
        f"{DATASET_LABELS[dataset]} n=50: Spearman rho with SE",
        fontsize=10.2,
        y=0.97,
    )
    ax.set_ylabel("Log-AURC", fontsize=9.0)
    ax.set_xlabel("Semivalue set", fontsize=9.0)
    ax.set_xticks(x)
    ax.set_xticklabels(family_names, fontsize=8.8)
    ax.set_ylim(*metric_limits(summary))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="y", labelsize=8.4)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=0.65,
            marker="_",
            markersize=6,
            label="SE",
        )
    )
    handles.append(
        Line2D([0], [0], color="#111111", linewidth=1.25, label="Worst")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.5,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.74)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Figure-1-style n=50 panel.")
    parser.add_argument("--dataset", choices=sorted(DATASET_LABELS), required=True)
    parser.add_argument("--budgets", default="10,20,30,40,50")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = split_ints(args.budgets)
    if len(budgets) < 2:
        raise ValueError("At least two budgets are required for Log-AURC.")
    if args.runs < 1:
        raise ValueError("runs must be positive.")
    results = run_matrix(args.dataset, budgets, args.runs)
    summary = summarize(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.dataset}_n50_figure_{stamp}"
    image_path = args.output_dir / f"{stem}.png"
    summary_path = args.output_dir / f"{stem}.csv"
    raw_path = args.output_dir / f"{stem}_raw.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    results.to_csv(raw_path, index=False)
    plot(summary, args.dataset, image_path)
    print(summary.to_string(index=False))
    print(f"wrote: {image_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {raw_path}")


if __name__ == "__main__":
    main()
