"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman_rho(reference, estimate) -> float:
    rho = spearmanr(reference, estimate).correlation
    if rho is None or not np.isfinite(rho):
        return 0.0
    return float(rho)


def pairwise_accuracy(reference, estimate) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    total = 0
    correct = 0
    for i in range(len(reference)):
        for j in range(i + 1, len(reference)):
            ref_order = np.sign(reference[i] - reference[j])
            est_order = np.sign(estimate[i] - estimate[j])
            if ref_order == 0:
                continue
            total += 1
            correct += int(ref_order == est_order)
    return float(correct / total) if total else 0.0


def topk_overlap(reference, estimate, k: int | None = None) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if k is None:
        k = max(1, int(round(0.2 * len(reference))))
    ref_top = set(np.argsort(reference)[-k:])
    est_top = set(np.argsort(estimate)[-k:])
    return float(len(ref_top & est_top) / k)


def log_aurc(curve: pd.DataFrame, budget_col: str = "uep", score_col: str = "rho") -> float:
    curve = curve.sort_values(budget_col)
    budgets = curve[budget_col].to_numpy(dtype=float)
    scores = curve[score_col].to_numpy(dtype=float)
    if len(budgets) < 2:
        return float(scores[0]) if len(scores) else 0.0
    log_b = np.log(budgets)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(scores, x=log_b)
    else:
        area = np.trapz(scores, x=log_b)
    return float(area / (log_b[-1] - log_b[0]))


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["method", "uep"], as_index=False)
        .agg(
            rho_mean=("rho", "mean"),
            rho_worst=("rho", "min"),
            pairwise_mean=("pairwise", "mean"),
            topk_mean=("topk", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .sort_values(["method", "uep"])
    )
    return grouped


def summarize_log_aurc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = df[df["status"] == "ok"].copy()
    for (method, semivalue), group in ok.groupby(["method", "semivalue"]):
        by_budget = group.groupby("uep", as_index=False).agg(rho=("rho", "mean"))
        rows.append(
            {
                "method": method,
                "semivalue": semivalue,
                "log_aurc": log_aurc(by_budget),
                "num_budget_points": int(by_budget["uep"].nunique()),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    return (
        detail.groupby("method", as_index=False)
        .agg(
            log_aurc_mean=("log_aurc", "mean"),
            log_aurc_worst=("log_aurc", "min"),
            num_semivalues=("semivalue", "nunique"),
        )
        .sort_values("log_aurc_mean", ascending=False)
    )
