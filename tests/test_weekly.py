"""Phase 5: the one weekly command, the CLI wiring and the README charts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import fplopt.pipeline as pipeline
from fplopt import cli
from fplopt.charts import differential_numbers, make_charts


@pytest.fixture
def tmp_cfg(cfg, tmp_path):
    paths = replace(cfg.paths, processed=tmp_path / "data" / "processed", models=tmp_path / "models",
                    reports=tmp_path / "reports", api_cache=tmp_path / "cache", snapshots=tmp_path / "snap")
    paths.processed.mkdir(parents=True)
    paths.reports.mkdir()
    return replace(cfg, paths=paths)


def predictions(gw: int = 9) -> pd.DataFrame:
    return pd.DataFrame({
        "gw": gw, "web_name": ["A", "B", "C"], "team_short": "T", "position": "MID",
        "now_cost": [55, 60, 70], "ownership_now": [2.0, 4.0, 30.0], "pred_points": [3.0, 5.0, 6.0],
        "p_haul": [0.1, 0.2, 0.3], "reasons": "r", "differential_score": [3.0, 5.0, np.nan],
    })


def test_fresh_client_ignores_cached_season_data(cfg):
    fresh, cached = pipeline.make_client(cfg, fresh=True), pipeline.make_client(cfg)
    for key in ("bootstrap", "fixtures", "element_summary"):
        assert fresh.ttl[key] == 0 and cached.ttl[key] > 0
    assert fresh.ttl["entry"] == cached.ttl["entry"]          # your team is always short-lived anyway


def test_weekly_summary_lists_differentials_and_team(tmp_cfg):
    text = pipeline.weekly_summary(predictions(), tmp_cfg, team_text="XI goes here")
    assert text.startswith("# GW9 weekly run")
    assert text.index("| B |") < text.index("| A |")              # ranked by differential score
    assert "| C |" not in text                                    # 30% owned: not a differential
    assert "XI goes here" in text


def test_weekly_runs_every_step_in_order(tmp_cfg, monkeypatch):
    calls = []
    table = pd.DataFrame({"season": ["s"], "y_points": [1.0]})
    monkeypatch.setattr("fplopt.data.vaastav.download_history", lambda *a, **k: calls.append("history"))
    monkeypatch.setattr(pipeline, "build_features", lambda cfg, live, client: calls.append("features") or table)
    monkeypatch.setattr("fplopt.baselines.baseline_report", lambda t, cfg: calls.append("baselines") or "report")
    monkeypatch.setattr(pipeline, "run_training", lambda cfg: calls.append("train"))
    monkeypatch.setattr(pipeline, "predict_upcoming", lambda cfg: calls.append("predict") or predictions())
    monkeypatch.setattr("fplopt.optimize.service.optimize_team",
                        lambda cfg, client, team_id: calls.append(f"optimize {team_id}") or {})
    monkeypatch.setattr("fplopt.optimize.render.render", lambda result: "plan text")

    summary, path = pipeline.run_weekly(tmp_cfg, team_id=42, client=object())
    assert calls == ["history", "features", "baselines", "train", "predict", "optimize 42"]
    assert path.name == "gw9.md" and path.read_text() == summary and "plan text" in summary
    assert path.parent == tmp_cfg.paths.processed.parent / "weekly"      # under data/: never committed

    calls.clear()
    (tmp_cfg.paths.models / "final").mkdir(parents=True)
    (tmp_cfg.paths.models / "final" / "models.joblib").write_text("x")
    pipeline.run_weekly(tmp_cfg, retrain=False, client=object())
    assert "train" not in calls and not any(c.startswith("optimize") for c in calls)


def test_cli_parses_the_weekly_command(monkeypatch):
    seen = {}
    monkeypatch.setitem(cli.COMMANDS, "weekly", (lambda cfg, args: seen.update(vars(args)), "help"))
    cli.main(["weekly", "--team-id", "123", "--no-retrain"])
    assert seen["team_id"] == 123 and seen["no_retrain"] and not seen["cached"]
    with pytest.raises(SystemExit):
        cli.main(["no-such-command"])


def test_charts_are_drawn_from_saved_results(tmp_cfg, tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for season, split in (("2024-25", "dev"), ("2025-26", "test"), ("2026-27", "test")):
        for gw in (2, 3):
            for code in range(60):
                rows.append({"season": season, "gw": gw, "code": code, "split": split, "position": "MID",
                             "price": 5.0, "ownership_pct": rng.uniform(0, 20), "n_fixtures": 1, "fix3_n": 3,
                             "pts_r3": rng.uniform(0, 5), "lgbm_points": rng.uniform(0, 6),
                             "ridge_points": rng.uniform(0, 6), "lgbm_haul": rng.uniform(0, 0.3),
                             "implied_points": 2.0, "xp_fpl": np.nan, "y_points": float(rng.integers(0, 12))})
    oos = pd.DataFrame(rows)
    numbers = differential_numbers(oos, tmp_cfg)
    assert numbers["label"].tolist() == ["LightGBM model", "Ridge model", "Last-3 average", "Any differential"]
    assert numbers["gameweeks"].iloc[0] == 4                       # test seasons only

    oos.to_parquet(tmp_cfg.paths.oos_predictions)
    weeks = pd.DataFrame([{"season": s, "source": src, "gw": gw, "points": 50.0}
                          for s in ("2024-25", "2025-26") for src in ("model", "last3") for gw in range(2, 39)])
    weeks.to_parquet(tmp_cfg.paths.processed / "backtest_weeks.parquet")
    paths = make_charts(tmp_cfg, out_dir=tmp_path / "img")
    assert sorted(p.name for p in paths) == ["backtest-dark.png", "backtest-light.png",
                                             "differentials-dark.png", "differentials-light.png"]
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in paths)
