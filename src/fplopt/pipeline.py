"""Glue between configuration, data sources, features and models (what the CLI calls)."""
from __future__ import annotations

import logging

import pandas as pd

from fplopt.config import Config
from fplopt.data.api import FPLClient
from fplopt.data.live import LiveState, load_live_season
from fplopt.data.schema import SeasonData
from fplopt.data.vaastav import load_season
from fplopt.features.build import build_feature_table

log = logging.getLogger(__name__)

FINAL_MODEL = "final"
# Reasons explain each player relative to a typical regular starter (60+ minutes a match, last 3).
REGULAR_STARTER_MINUTES = 60


def make_client(cfg: Config) -> FPLClient:
    return FPLClient(
        cache_dir=cfg.paths.api_cache,
        snapshot_dir=cfg.paths.snapshots,
        base_url=cfg.live.base_url,
        ttl=cfg.live.ttl_seconds,
        workers=cfg.live.request_workers,
    )


def load_seasons(cfg: Config, live: bool = True,
                 client: FPLClient | None = None) -> tuple[list[SeasonData], LiveState | None]:
    """Historical seasons from disk, plus the current season from the API when ``live``."""
    seasons = [load_season(s, cfg.paths.raw_history, cfg.history.xp_valid_min_nonzero_share)
               for s in cfg.history.seasons]
    state = None
    if live:
        season, state = load_live_season(client or make_client(cfg), cfg.live.season)
        seasons.append(season)
    return seasons, state


def build_features(cfg: Config, live: bool = True, client: FPLClient | None = None) -> pd.DataFrame:
    """Build and save the feature table, including rows for the next gameweek when live.

    When live, also saves the current player list with availability flags, price
    and ownership right now (``live_players.parquet``), which serving needs.
    """
    seasons, state = load_seasons(cfg, live=live, client=client)
    upcoming = {state.season: [state.next_gw]} if state and state.next_gw else None
    table = build_feature_table(seasons, cfg, upcoming=upcoming)
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cfg.paths.features, index=False)
    log.info("Wrote %s rows to %s", f"{len(table):,}", cfg.paths.features)
    if state:
        current = seasons[-1]
        players = current.players.merge(current.teams[["team_id", "team_short"]], on="team_id", how="left")
        players.assign(season=state.season, next_gw=state.next_gw).to_parquet(cfg.paths.live_players, index=False)
    return table


def run_training(cfg: Config) -> str:
    """Walk-forward evaluation, weight selection on dev, final models, report. Returns the report."""
    from fplopt.models.dataset import design_matrix
    from fplopt.models.differential import choose_weights
    from fplopt.models.explain import global_importance
    from fplopt.models.report import evaluation_frame, model_report
    from fplopt.models.train import fit_final, walk_forward

    table = pd.read_parquet(cfg.paths.features)
    oos = walk_forward(table, cfg)
    oos.to_parquet(cfg.paths.oos_predictions, index=False)

    dev = oos[oos["split"] == "dev"].assign(pred_points=lambda d: d["lgbm_points"], p_haul=lambda d: d["lgbm_haul"])
    weights, grid = choose_weights(dev, cfg.differential.grid, cfg.evaluation.top_k,
                                   cfg.differential.ownership_max_pct, cfg.targets.return_points,
                                   cfg.targets.haul_points)
    log.info("Differential weights chosen on dev: %s", weights)

    final = fit_final(table, cfg, weights)
    final.save(cfg.paths.models / FINAL_MODEL)

    recent_seasons = sorted(table["season"].unique())[-2:]
    recent = table[~table["is_future"] & table["season"].isin(recent_seasons)]
    X = design_matrix(recent, cfg)
    importance = {name: global_importance(final.models[f"lgbm_{name}"].contributions(X), list(X.columns))
                  for name in ("points", "haul")}

    report = model_report(evaluation_frame(oos, weights, cfg), weights, grid, importance,
                          final.meta["lgbm_trees"], cfg)
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "phase2_model.md").write_text(report)
    return report


def availability(players: pd.DataFrame) -> pd.Series:
    """Chance of playing as a 0-1 multiplier: FPL's percentage if set, else 1 if available, 0 if not."""
    flagged = players["chance_of_playing_next_round"] / 100
    return flagged.where(flagged.notna(), (players["status"] == "a").astype("float64"))


def predict_upcoming(cfg: Config) -> pd.DataFrame:
    """Predictions for the next gameweek with the final models, live availability and reasons."""
    from fplopt.models.dataset import design_matrix
    from fplopt.models.differential import differential_score
    from fplopt.models.explain import top_reasons
    from fplopt.models.train import ModelBundle, predict_frame

    table = pd.read_parquet(cfg.paths.features)
    future = table[table["is_future"]].reset_index(drop=True)
    if future.empty:
        raise RuntimeError("No upcoming gameweek rows - run `fplopt build-features` (live) first")
    bundle = ModelBundle.load(cfg.paths.models / FINAL_MODEL)
    h = cfg.targets.multi_gw_horizon

    out = predict_frame(bundle, future, cfg)
    X = design_matrix(future, cfg)
    contribs = bundle.models["lgbm_points"].contributions(X)
    starters = (future["mins_r3"] >= REGULAR_STARTER_MINUTES).to_numpy()
    reference = contribs[starters, :-1].mean(axis=0) if starters.any() else None
    out["reasons"] = top_reasons(contribs, X, reference=reference)

    live = pd.read_parquet(cfg.paths.live_players)
    keep = ["code", "status", "chance_of_playing_next_round", "news", "now_cost", "selected_by_percent", "ep_next"]
    out = out.merge(live[keep], on="code", how="left")
    out["availability"] = availability(out)
    out["pred_points"] = out["lgbm_points"] * out["availability"]
    out["p_haul"] = out["lgbm_haul"] * out["availability"]
    out[f"pred_next{h}"] = out[f"lgbm_next{h}"]
    out["edge"] = out["pred_points"] - out["implied_points"]
    # eligibility uses ownership right now; the score's inputs stay as the model was trained
    out["ownership_now"] = out["selected_by_percent"].where(out["selected_by_percent"].notna(), out["ownership_pct"])
    out["differential_score"] = differential_score(out, bundle.meta["differential_weights"],
                                                   cfg.differential.ownership_max_pct, ownership="ownership_now")
    out = out.sort_values("pred_points", ascending=False, ignore_index=True)
    out.to_parquet(cfg.paths.predictions, index=False)
    return out


def top_differentials(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    flagged = predictions.dropna(subset=["differential_score"])
    return flagged.sort_values("differential_score", ascending=False).head(k).reset_index(drop=True)
