"""Feature values checked by hand against the synthetic league (see tests/synthetic.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplopt.features.tables import player_matches
from synthetic import LATE_SIGNING

S1, S2 = "2022-23", "2023-24"
PLAYER = 103          # team 1 midfielder: team 1 has a double in GW5 and a blank in GW6


@pytest.fixture(scope="module")
def matches(league):
    m = pd.concat([player_matches(s) for s in league], ignore_index=True)
    return m.sort_values(["code", "season", "gw", "kickoff_time"], ignore_index=True)


def row(table, season, gw, code=PLAYER):
    r = table[(table["season"] == season) & (table["gw"] == gw) & (table["code"] == code)]
    assert len(r) == 1, f"expected one row for {code} in {season} GW{gw}, found {len(r)}"
    return r.iloc[0]


def before(matches, season, gw, code=PLAYER):
    m = matches[matches["code"] == code]
    return m[(m["season"] < season) | ((m["season"] == season) & (m["gw"] < gw))]


@pytest.mark.parametrize("season, gw", [(S2, 1), (S2, 4), (S2, 7)])
def test_rolling_form_averages_the_previous_fixtures(table, matches, season, gw):
    # GW1 of season 2 reaches back into season 1 (via the stable player code); GW7 skips the blank GW6
    prior = before(matches, season, gw)
    r = row(table, season, gw)
    assert r["pts_r3"] == pytest.approx(prior["total_points"].tail(3).mean())
    assert r["mins_r5"] == pytest.approx(prior["minutes"].tail(5).mean())
    assert r["xg_r5"] == pytest.approx(prior["expected_goals"].tail(5).mean())
    assert r["mins_last"] == prior["minutes"].iloc[-1]


def test_double_gameweek_sums_both_fixtures(table, matches):
    r = row(table, S2, 5)
    gw5 = matches[(matches["code"] == PLAYER) & (matches["season"] == S2) & (matches["gw"] == 5)]
    assert len(gw5) == 2
    assert r["n_fixtures"] == 2
    assert r["y_points"] == gw5["total_points"].sum()


def test_blank_gameweek_has_no_row(table):
    assert table[(table["season"] == S2) & (table["gw"] == 6) & (table["code"] == PLAYER)].empty


def test_market_features_come_from_the_previous_gameweek(table, matches):
    m = matches[(matches["code"] == PLAYER) & (matches["season"] == S2)].groupby("gw").first()
    r4 = row(table, S2, 4)
    assert r4["price"] == m.loc[3, "value"] / 10
    assert r4["price_change_1gw"] == pytest.approx((m.loc[3, "value"] - m.loc[2, "value"]) / 10)
    assert r4["price_change_season"] == pytest.approx((m.loc[3, "value"] - m.loc[1, "value"]) / 10)
    assert row(table, S2, 7)["price"] == m.loc[5, "value"] / 10      # previous row is GW5 (blank in GW6)
    assert np.isnan(row(table, S2, 1)["price"])                       # nothing earlier this season


def test_ownership_is_a_share_of_all_managers(table, matches):
    gw3 = matches[(matches["season"] == S2) & (matches["gw"] == 3)].drop_duplicates("code")
    managers = gw3["selected"].sum() / 15                              # every team plays in GW3
    selected = gw3.loc[gw3["code"] == PLAYER, "selected"].iloc[0]
    assert row(table, S2, 4)["ownership_pct"] == pytest.approx(selected / managers * 100)


def test_promoted_team_starts_from_the_prior(table, cfg):
    prior = cfg.features.promoted_prior
    promoted = table[(table["season"] == S2) & (table["gw"] == 1) & (table["team_code"] == 5)]
    established = table[(table["season"] == S2) & (table["gw"] == 1) & (table["team_code"] == 1)]
    assert not promoted.empty
    assert (promoted["team_xg_for"] == prior["xg_for"]).all()
    assert (promoted["team_points"] == prior["points"]).all()
    assert (established["team_xg_for"] != prior["xg_for"]).all()


def test_opponent_strength_matches_the_opponents_own_rows(table):
    """A team's opp_* features are the opponent's team_* features for the same gameweek."""
    r = row(table, S2, 4)
    sched_opp = table[(table["season"] == S2) & (table["gw"] == 4) & (table["team_code"] != r["team_code"])]
    candidates = sched_opp.groupby("team_code")[["team_xg_against", "team_points"]].first()
    match = candidates[(candidates["team_xg_against"] == r["opp_xg_against"])
                       & (candidates["team_points"] == r["opp_points"])]
    assert len(match) == 1


def test_fixture_horizon_counts_doubles_and_blanks(table):
    assert row(table, S2, 4)["fix3_n"] == 3     # GW4 (1) + GW5 (2) + GW6 (0)
    assert row(table, S2, 5)["fix3_n"] == 3     # GW5 (2) + GW6 (0) + GW7 (1)
    assert row(table, S2, 7)["fix3_n"] == 2     # GW7 + GW8; the season ends


def test_late_signing_has_no_form_before_his_first_match(table, matches):
    code, first = LATE_SIGNING["code"], LATE_SIGNING["from_gw"]
    debut = row(table, S2, first, code)
    assert np.isnan(debut["pts_r3"]) and np.isnan(debut["price"])
    played = before(matches, S2, first + 1, code)
    assert row(table, S2, first + 1, code)["pts_r3"] == pytest.approx(played["total_points"].mean())


def test_targets(table, matches, cfg):
    pts = matches[(matches["code"] == PLAYER) & (matches["season"] == S2)].groupby("gw")["total_points"].sum()
    r4 = row(table, S2, 4)
    assert r4["y_points_next3"] == pts[4] + pts[5]                # GW6 is a blank: 0 points
    assert r4["y_haul"] == float(pts[4] >= cfg.targets.haul_points)
    assert np.isnan(row(table, S2, 7)["y_points_next3"])          # would need GW9
