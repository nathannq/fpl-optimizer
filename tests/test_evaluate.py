"""Evaluation metrics and baselines on tiny hand-made tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplopt.baselines import add_baselines
from fplopt.evaluate import differential_metrics, error_metrics, top_k_differentials


@pytest.fixture
def gameweek():
    # one gameweek, six players; P5 is owned by 40% so is not a differential
    return pd.DataFrame({
        "season": "s", "gw": 1, "code": [1, 2, 3, 4, 5, 6],
        "score": [9.0, 8.0, 7.0, 1.0, 10.0, np.nan],
        "ownership_pct": [1.0, 2.0, 3.0, 4.0, 40.0, 5.0],
        "y_points": [10.0, 2.0, 5.0, 0.0, 12.0, 1.0],
    })


def test_top_k_respects_ownership_threshold_and_k(gameweek):
    picks = top_k_differentials(gameweek, "score", k=2, ownership_max_pct=10)
    assert picks["code"].tolist() == [1, 2]


def test_differential_metrics(gameweek):
    m = differential_metrics(gameweek, "score", k=3, ownership_max_pct=10, return_points=5, haul_points=8)
    assert m["precision@3"] == pytest.approx(2 / 3)     # 10 and 5 points returned; 2 did not
    assert m["haul_rate"] == pytest.approx(1 / 3)
    assert m["mean_points"] == pytest.approx(17 / 3)
    assert m["pool_mean_points"] == pytest.approx(18 / 5)   # all five differentials, scored or not


def test_error_metrics():
    df = pd.DataFrame({"season": "s", "gw": [1, 1, 1, 2, 2, 2], "pred": [1.0, 2, 3, 1, 2, 3],
                       "y_points": [1.0, 2, 3, 3, 2, 1]})
    m = error_metrics(df, "pred")
    assert m["mae"] == pytest.approx(4 / 6)
    assert m["bias"] == pytest.approx(0)
    assert m["spearman"] == pytest.approx(0)            # +1 in GW1, -1 in GW2


def test_last3_baseline_scales_with_fixtures_and_defaults_to_zero():
    df = pd.DataFrame({"pts_r3": [2.0, 2.0, np.nan], "n_fixtures": [1, 2, 1], "xp_fpl": [1.0, 2.0, 3.0]})
    out = add_baselines(df)
    assert out["base_last3"].tolist() == [2.0, 4.0, 0.0]
    assert out["base_fpl_xp"].tolist() == [1.0, 2.0, 3.0]
