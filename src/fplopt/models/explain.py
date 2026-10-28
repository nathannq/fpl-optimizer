"""Readable explanations from per-row SHAP values (LightGBM's built-in TreeSHAP)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

STAT_LABELS = {
    "pts": "points", "mins": "minutes", "starts": "starts", "goals": "goals", "assists": "assists",
    "xg": "xG", "xa": "xA", "xgi": "xGI", "bonus": "bonus", "bps": "BPS", "cs": "clean sheets",
    "defcon": "defensive contributions",
}
FIXED_LABELS = {
    "mins_last": "minutes last match",
    "mins_trend": "minutes trend",
    "matches_since_start": "matches since last start",
    "played_last": "played last match",
    "pts_season_avg": "points per match this season",
    "matches_season": "matches this season",
    "price": "price",
    "price_change_1gw": "price change last GW",
    "price_change_season": "price change this season",
    "ownership_pct": "ownership",
    "ownership_change_1gw": "ownership change last GW",
    "net_transfers_pct": "net transfers last GW",
    "n_fixtures": "fixtures this GW",
    "home_frac": "at home",
    "position": "position",
}
TEAM_STAT_LABELS = {
    "xg_for": "xG per match", "xg_against": "xG conceded per match",
    "goals_for": "goals per match", "goals_against": "goals conceded per match",
    "points": "points per match", "clean_sheet": "clean-sheet rate",
}


def describe(feature: str) -> str:
    """'xgi_r5' -> 'xGI per match, last 5'; 'team_xg_for' -> 'team xG for'; etc."""
    if feature in FIXED_LABELS:
        return FIXED_LABELS[feature]
    if m := re.fullmatch(r"(\w+?)_p90_r(\d+)", feature):
        return f"{STAT_LABELS.get(m[1], m[1])} per 90, last {m[2]}"
    if m := re.fullmatch(r"(\w+?)_r(\d+)", feature):
        return f"{STAT_LABELS.get(m[1], m[1])} per match, last {m[2]}"
    if m := re.fullmatch(r"fix(\d+)_(.+)", feature):
        rest = {"n": "fixtures", "opp_xg_against": "opponents' xG conceded",
                "opp_xg_for": "opponents' xG"}.get(m[2], m[2])
        return f"next {m[1]} GWs: {rest}"
    for prefix, who in (("team_", "team"), ("opp_", "opponent")):
        if feature.startswith(prefix) and feature[len(prefix):] in TEAM_STAT_LABELS:
            return f"{who} {TEAM_STAT_LABELS[feature[len(prefix):]]}"
    return feature


def format_value(feature: str, value) -> str:
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "n/a"
    if feature == "price":
        return f"£{value:.1f}m"
    if feature in ("ownership_pct", "ownership_change_1gw", "net_transfers_pct"):
        return f"{value:+.1f}%" if "change" in feature or "net" in feature else f"{value:.1f}%"
    if feature == "home_frac":
        return "yes" if value >= 1 else "no" if value <= 0 else f"{value:.0%}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def top_reasons(contribs: np.ndarray, X: pd.DataFrame, n: int = 3,
                reference: np.ndarray | None = None) -> list[str]:
    """For each row, the ``n`` features pushing the prediction up the most.

    ``contribs`` is LightGBM ``pred_contrib`` output (features..., base value).
    SHAP values are relative to the average row, and most rows are players who
    don't play, so "plays 90 minutes" would top every starter's list. Passing
    ``reference`` (mean contributions over a comparison group, e.g. regular
    starters) explains each player relative to that group instead; SHAP values
    are additive, so this is a shift of the baseline, not an approximation.
    """
    feats = list(X.columns)
    values = X.to_numpy(dtype=object)
    shap = contribs[:, :-1] - (0 if reference is None else reference)
    reasons = []
    for i, row in enumerate(shap):
        order = [j for j in np.argsort(-row) if row[j] > 0][:n]
        reasons.append("; ".join(
            f"{describe(feats[j])} {format_value(feats[j], values[i, j])} (+{row[j]:.1f})" for j in order))
    return reasons


def global_importance(contribs: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature: how much each moves predictions on average."""
    imp = np.abs(contribs[:, :-1]).mean(axis=0)
    out = pd.DataFrame({"feature": features, "description": [describe(f) for f in features], "mean_abs_shap": imp})
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
