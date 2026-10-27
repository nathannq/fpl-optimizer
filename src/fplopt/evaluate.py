"""Evaluation shared by the baselines (Phase 1) and the model (Phase 2).

Two questions are asked of any projection:

1. How accurate is it? -> MAE / RMSE / bias, and the mean per-gameweek
   Spearman rank correlation (ranking is what transfer decisions use).
2. Does it find differentials? -> each gameweek, take the top-K players under
   the ownership threshold by the projection and check how they actually did:
   precision@K (share that returned >= ``return_points``), haul rate and mean
   points, against the average differential in the same pool.

Ownership is the value going *into* the gameweek (``ownership_pct`` feature),
i.e. what you would have seen when choosing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def error_metrics(df: pd.DataFrame, pred: str, target: str = "y_points") -> dict[str, float]:
    d = df[[pred, target, "season", "gw"]].dropna()
    err = d[pred] - d[target]
    per_gw = d.groupby(["season", "gw"])[[pred, target]].corr(method="spearman").xs(target, level=2)[pred]
    return {
        "n": float(len(d)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "spearman": float(per_gw.mean()),
    }


def differential_pool(df: pd.DataFrame, ownership_max_pct: float) -> pd.DataFrame:
    """Players under the ownership threshold, with a known outcome."""
    return df[(df["ownership_pct"] < ownership_max_pct) & df["y_points"].notna()]


def top_k_differentials(df: pd.DataFrame, score: str, k: int, ownership_max_pct: float) -> pd.DataFrame:
    """The ``k`` highest-scoring differentials in each gameweek (ties broken by player code)."""
    pool = differential_pool(df, ownership_max_pct).dropna(subset=[score])
    pool = pool.sort_values(["season", "gw", score, "code"], ascending=[True, True, False, True])
    return pool.groupby(["season", "gw"], sort=False).head(k)


def differential_metrics(df: pd.DataFrame, score: str, k: int, ownership_max_pct: float,
                         return_points: int, haul_points: int) -> dict[str, float]:
    picks = top_k_differentials(df, score, k, ownership_max_pct)
    pool = differential_pool(df, ownership_max_pct)
    return {
        "gameweeks": float(picks.groupby(["season", "gw"]).ngroups),
        f"precision@{k}": float((picks["y_points"] >= return_points).mean()),
        "haul_rate": float((picks["y_points"] >= haul_points).mean()),
        "mean_points": float(picks["y_points"].mean()),
        "pool_mean_points": float(pool["y_points"].mean()),
        "pool_return_rate": float((pool["y_points"] >= return_points).mean()),
    }
