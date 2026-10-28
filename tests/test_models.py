"""Phase 2: folds, reproducibility, out-of-sample discipline, differential score, explanations."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fplopt.models.dataset import design_matrix, early_stopping_split, walk_forward_folds
from fplopt.models.differential import MarketModel, choose_weights, differential_score
from fplopt.models.explain import describe, top_reasons
from fplopt.models.report import paired_bootstrap
from fplopt.models.train import ModelBundle, fit_bundle, predict_frame, walk_forward
from fplopt.pipeline import availability

S1, S2 = "2022-23", "2023-24"


@pytest.fixture(scope="module")
def fast_cfg(cfg):
    """Small LightGBM settings and the synthetic league's seasons."""
    lgbm = {**cfg.model.lightgbm, "n_estimators": 60, "min_child_samples": 10, "learning_rate": 0.1}
    return replace(cfg, model=replace(cfg.model, lightgbm=lgbm, early_stopping_rounds=10,
                                      dev_seasons=(), test_seasons=(S2,)))


def test_folds_only_train_on_earlier_seasons():
    folds = walk_forward_folds(["2022-23", "2023-24", "2024-25", "2025-26"], ["2025-26", "2024-25", "2022-23"])
    assert [(f.test_season, f.train_seasons) for f in folds] == [
        ("2024-25", ("2022-23", "2023-24")),
        ("2025-26", ("2022-23", "2023-24", "2024-25")),
    ]   # 2022-23 has nothing earlier, so it is never a test season


def test_early_stopping_uses_the_latest_rows():
    two = pd.DataFrame({"season": ["a", "a", "b", "b"], "gw": [1, 2, 1, 2]})
    assert early_stopping_split(two).tolist() == [False, False, True, True]
    one = pd.DataFrame({"season": "a", "gw": range(1, 9)})
    assert early_stopping_split(one).tolist() == [False] * 6 + [True] * 2


def test_training_is_reproducible(table, fast_cfg):
    rows = table[table["season"] == S1]
    a = predict_frame(fit_bundle(rows, fast_cfg), table, fast_cfg)
    b = predict_frame(fit_bundle(rows, fast_cfg), table, fast_cfg)
    pd.testing.assert_frame_equal(a, b)


def test_out_of_sample_predictions_ignore_test_season_outcomes(table, fast_cfg):
    """Scrambling every target in the test season must not change its predictions."""
    scrambled = table.copy()
    test = scrambled["season"] == S2
    rng = np.random.default_rng(0)
    for col in ("y_points", "y_points_next3"):
        scrambled.loc[test, col] = rng.integers(0, 20, test.sum()).astype(float)
    scrambled.loc[test, "y_haul"] = (scrambled.loc[test, "y_points"] >= 8).astype(float)
    a, b = walk_forward(table, fast_cfg), walk_forward(scrambled, fast_cfg)
    pred_cols = [c for c in a.columns if c.startswith(("ridge_", "lgbm_")) or c == "implied_points"]
    assert set(a["season"]) == {S2}
    pd.testing.assert_frame_equal(a[pred_cols], b[pred_cols])


def test_saved_bundle_predicts_the_same(table, fast_cfg, tmp_path):
    bundle = fit_bundle(table[table["season"] == S1], fast_cfg)
    bundle.meta["differential_weights"] = {"haul_weight": 0.0, "edge_weight": 0.0, "price_weight": 0.0}
    bundle.save(tmp_path / "m")
    loaded = ModelBundle.load(tmp_path / "m")
    pd.testing.assert_frame_equal(predict_frame(bundle, table, fast_cfg), predict_frame(loaded, table, fast_cfg))
    assert loaded.meta["differential_weights"]["haul_weight"] == 0.0


def scored_frame():
    return pd.DataFrame({
        "pred_points": [5.0, 4.0, 3.0, 6.0],
        "p_haul": [0.1, 0.5, 0.2, 0.3],
        "implied_points": [4.0, 1.0, 3.0, 5.0],
        "price": [8.0, 5.0, 4.5, 12.0],
        "ownership_pct": [5.0, 2.0, 9.9, 30.0],
    })


def test_differential_score_formula_and_ownership_threshold():
    df = scored_frame()
    w = {"haul_weight": 2.0, "edge_weight": 0.5, "price_weight": 0.1}
    s = differential_score(df, w, ownership_max_pct=10)
    assert s[0] == pytest.approx(5.0 + 0.2 + 0.5 - 0.8)
    assert np.isnan(s[3])                                     # 30% owned: not a differential
    assert differential_score(df, w, ownership_max_pct=50).notna().all()   # threshold is configurable


def test_weights_are_chosen_on_dev_precision():
    # Within each gameweek, P(haul) identifies who returns while predicted points do not.
    rng = np.random.default_rng(1)
    rows = []
    for gw in range(1, 21):
        for i in range(30):
            hauls = i < 5
            rows.append({"season": "dev", "gw": gw, "code": i, "ownership_pct": 1.0, "price": 5.0,
                         "pred_points": rng.uniform(2, 3), "implied_points": 2.0,
                         "p_haul": 0.6 if hauls else 0.05, "y_points": 10.0 if hauls else 1.0})
    weights, table = choose_weights(pd.DataFrame(rows), {"haul_weight": [0, 5], "edge_weight": [0],
                                                         "price_weight": [0]}, k=5, ownership_max_pct=10,
                                    return_points=5, haul_points=8)
    assert weights["haul_weight"] == 5
    assert table.loc[0, "precision"] == 1.0


def test_market_model_needs_ownership_and_price(table):
    market = MarketModel().fit(table[table["season"] == S1])
    preds = market.predict(table)
    missing = table["ownership_pct"].isna() | table["price"].isna()
    assert np.isnan(preds[missing.to_numpy()]).all()
    assert np.isfinite(preds[~missing.to_numpy()]).all()


def test_reasons_can_be_relative_to_a_reference_group():
    X = pd.DataFrame({"xgi_r5": [0.6, 0.1], "mins_last": [90.0, 90.0]})
    contribs = np.array([[0.5, 1.5, 1.0], [0.0, 1.5, 1.0]])   # features..., base value
    plain = top_reasons(contribs, X, n=1)
    assert plain[0].startswith("minutes last match 90")
    relative = top_reasons(contribs, X, n=1, reference=np.array([0.0, 1.5]))
    assert relative == ["xGI per match, last 5 0.6 (+0.5)", ""]


def test_feature_descriptions():
    assert describe("xg_p90_r5") == "xG per 90, last 5"
    assert describe("fix3_opp_xg_against") == "next 3 GWs: opponents' xG conceded"
    assert describe("opp_xg_for") == "opponent xG per match"
    assert describe("team_clean_sheet") == "team clean-sheet rate"


def test_availability_from_live_flags():
    live = pd.DataFrame({"chance_of_playing_next_round": [75.0, np.nan, np.nan, 0.0],
                         "status": ["d", "a", "i", "s"]})
    assert availability(live).tolist() == [0.75, 1.0, 0.0, 0.0]


def test_paired_bootstrap():
    a = pd.DataFrame({"m": [0.5, 0.4, 0.6, 0.5]})
    b = pd.DataFrame({"m": [0.3, 0.4, 0.2, 0.5]})
    r = paired_bootstrap(a, b, "m", n_boot=2000, seed=0)
    assert r["diff"] == pytest.approx(0.15)
    assert r["ci_low"] <= r["diff"] <= r["ci_high"]
    assert r["gw_wins"] == 0.5 and r["gw_ties"] == 0.5
