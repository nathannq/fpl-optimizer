"""The differential score: predicted value above what ownership implies, for low-owned players.

    implied_points = what the market alone predicts: ownership, price, position, fixtures
    edge           = pred_points - implied_points
    score          = pred_points + haul_weight * P(haul) + edge_weight * edge - price_weight * price

Only players owned by fewer than ``ownership_max_pct`` get a score. The weights
are chosen by :func:`choose_weights` on the dev seasons, never on test seasons.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fplopt.evaluate import differential_metrics
from fplopt.models.dataset import POSITION_CATEGORIES

WEIGHT_NAMES = ("haul_weight", "edge_weight", "price_weight")


def _market_matrix(df: pd.DataFrame) -> pd.DataFrame:
    log_own = np.log1p(df["ownership_pct"])
    cols = {"log_own": log_own, "price": df["price"], "n_fixtures": df["n_fixtures"]}
    for pos in POSITION_CATEGORIES:
        is_pos = (df["position"] == pos).astype("float64")
        cols[f"is_{pos}"] = is_pos
        cols[f"log_own_{pos}"] = log_own * is_pos
        cols[f"price_{pos}"] = df["price"] * is_pos
    return pd.DataFrame(cols, index=df.index)


class MarketModel:
    """Expected points implied by the market alone (a linear fit, per position).

    This is the "what ownership implies" reference: a player the crowd owns
    heavily and FPL prices highly is expected to score more anyway.
    """

    def fit(self, df: pd.DataFrame) -> "MarketModel":
        rows = df.dropna(subset=["y_points", "ownership_pct", "price"])
        self.model_ = Ridge(alpha=1.0).fit(_market_matrix(rows), rows["y_points"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        ok = (df["ownership_pct"].notna() & df["price"].notna()).to_numpy()
        if ok.any():
            out[ok] = self.model_.predict(_market_matrix(df[ok]))
        return out


def differential_score(df: pd.DataFrame, weights: dict[str, float], ownership_max_pct: float,
                       points: str = "pred_points", haul: str = "p_haul",
                       implied: str = "implied_points", ownership: str = "ownership_pct") -> pd.Series:
    score = (df[points]
             + weights["haul_weight"] * df[haul]
             + weights["edge_weight"] * (df[points] - df[implied])
             - weights["price_weight"] * df["price"])
    return score.where(df[ownership] < ownership_max_pct)


def weight_grid(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    return [dict(zip(WEIGHT_NAMES, combo)) for combo in itertools.product(*(grid[n] for n in WEIGHT_NAMES))]


def choose_weights(dev: pd.DataFrame, grid: dict[str, list[float]], k: int, ownership_max_pct: float,
                   return_points: int, haul_points: int) -> tuple[dict[str, float], pd.DataFrame]:
    """Pick the weights with the best precision@k on ``dev`` (ties: higher mean points, then simpler).

    ``dev`` needs pred_points, p_haul, implied_points, price, ownership_pct and outcomes.
    """
    results = []
    for weights in weight_grid(grid):
        scored = dev.assign(score=differential_score(dev, weights, ownership_max_pct))
        m = differential_metrics(scored, "score", k, ownership_max_pct, return_points, haul_points)
        results.append({**weights, "precision": m[f"precision@{k}"], "haul_rate": m["haul_rate"],
                        "mean_points": m["mean_points"], "complexity": sum(w != 0 for w in weights.values())})
    table = pd.DataFrame(results).sort_values(["precision", "mean_points", "complexity"],
                                              ascending=[False, False, True], ignore_index=True)
    best = {n: float(table.loc[0, n]) for n in WEIGHT_NAMES}
    return best, table.drop(columns="complexity")
