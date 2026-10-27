"""The no-leakage rule: a feature for gameweek N may only use information from before N.

The main test is a perturbation test. Rewrite, at random, every result and
market value from gameweek N onwards (points, minutes, goals, xG, bonus, prices,
ownership, team scores...), rebuild the table, and require every feature on every
row up to and including gameweek N to be bit-for-bit unchanged. Only the fixture
schedule is kept, as it is published in advance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fplopt.data.schema import MARKET_COLUMNS, MATCH_STATS
from fplopt.data.vaastav import load_season
from fplopt.features.build import CATEGORICAL_FEATURES, build_feature_table, feature_columns, target_columns
from synthetic import N_GWS, make_league, perturb_from

KEYS = ["season", "gw", "code"]
# Features that depend only on the published schedule or the player's position
# (matches_season counts the team's fixtures so far, not anything that happened in them).
SCHEDULE_ONLY = {"n_fixtures", "home_frac", "fix3_n", "matches_season", *CATEGORICAL_FEATURES}


def _rows_up_to(table: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    keep = (table["season"] < season) | ((table["season"] == season) & (table["gw"] <= gw))
    return table[keep].set_index(KEYS).sort_index()


def assert_no_leak(seasons, cfg, season: str, gw: int) -> None:
    before = build_feature_table(seasons, cfg)
    after = build_feature_table(perturb_from(seasons, season, gw), cfg)
    feats = feature_columns(cfg)
    a, b = _rows_up_to(before, season, gw), _rows_up_to(after, season, gw)
    pd.testing.assert_frame_equal(a[feats], b[feats], check_exact=True)
    # the perturbation really did change gameweek N's outcomes
    target_n = a.xs(gw, level="gw", drop_level=False)["y_points"]
    assert not target_n.equals(b.loc[target_n.index, "y_points"])


@pytest.mark.parametrize("gw", range(1, N_GWS + 1))
def test_features_ignore_results_from_gameweek_n_onwards(cfg, gw):
    assert_no_leak(make_league(seed=1), cfg, "2023-24", gw)


@pytest.mark.parametrize("gw", [1, 5, 6, 8])
def test_earlier_season_features_ignore_their_future(cfg, gw):
    """Perturbing season 1 from GW N onwards (and all of season 2) leaves season 1 up to N intact."""
    assert_no_leak(make_league(seed=2), cfg, "2022-23", gw)


def test_every_feature_responds_to_past_data(cfg):
    """Guard against the leak test passing vacuously: each non-schedule feature must
    change somewhere when results *before* the rows being compared change."""
    league = make_league(seed=3)
    before = build_feature_table(league, cfg).set_index(KEYS).sort_index()
    after = build_feature_table(perturb_from(league, "2023-24", 2), cfg).set_index(KEYS).sort_index()
    later = before.index[(before.index.get_level_values("season") == "2023-24")
                         & (before.index.get_level_values("gw") > 2)]
    unchanged = [f for f in feature_columns(cfg) if f not in SCHEDULE_ONLY
                 and before.loc[later, f].equals(after.loc[later, f])]
    assert unchanged == []


def test_live_rows_match_historical_rows(cfg):
    """A gameweek built as 'upcoming' (no results yet) gets exactly the features it gets in hindsight."""
    gw = 6
    hindsight = build_feature_table(make_league(seed=4), cfg)
    live = build_feature_table(make_league(seed=4, live_from_gw=gw), cfg, upcoming={"2023-24": [gw]})
    feats = feature_columns(cfg)
    h = hindsight[(hindsight["season"] == "2023-24") & (hindsight["gw"] == gw)].set_index("code").sort_index()
    lv = live[(live["season"] == "2023-24") & (live["gw"] == gw)].set_index("code").sort_index()
    assert lv["is_future"].all() and lv["y_points"].isna().all()
    pd.testing.assert_frame_equal(h.loc[lv.index, feats], lv[feats], check_exact=True)


def test_outcomes_are_never_features(cfg):
    feats = set(feature_columns(cfg))
    assert feats.isdisjoint(target_columns(cfg))
    assert "xp_fpl" not in feats            # recorded after the gameweek in the source data
    assert feats.isdisjoint(MATCH_STATS)    # raw same-gameweek stats
    assert feats.isdisjoint(MARKET_COLUMNS)  # raw same-gameweek market data


RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "vaastav"


@pytest.mark.slow
@pytest.mark.skipif(not (RAW / "2024-25").exists(), reason="historical data not downloaded")
def test_no_leak_on_real_data(cfg):
    seasons = [load_season(s, RAW, cfg.history.xp_valid_min_nonzero_share) for s in ("2023-24", "2024-25")]
    assert_no_leak(seasons, cfg, "2024-25", 20)
