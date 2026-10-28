"""FastAPI app: a JSON API over the pipeline plus a static single-page frontend.

    GET /                         the web UI
    GET /api/status               gameweek, deadline, data and model freshness
    GET /api/team/{team_id}       transfer options, XI, captain for a manager's team
    GET /api/squad                the best new squad within budget
    GET /api/differentials        top low-owned players with fixtures and reasons
    GET /api/league/{league_id}   mini-league standings and league ownership

Run with ``python -m fplopt serve``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fplopt.config import Config, load_config
from fplopt.data.api import FPLClient
from fplopt.optimize.projections import SOURCES
from fplopt.optimize.service import optimize_team
from fplopt.pipeline import FINAL_MODEL, make_client
from fplopt.web.views import differentials, league_view, merged_players, upcoming_fixtures

STATIC = Path(__file__).parent / "static"
Projection = Literal["model", "ep_next"]


class _ParquetCache:
    """Re-reads a parquet file only when it changes on disk (the weekly refresh rewrites them)."""

    def __init__(self):
        self._cache: dict[Path, tuple[float, pd.DataFrame]] = {}

    def read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        mtime = path.stat().st_mtime
        hit = self._cache.get(path)
        if hit is None or hit[0] != mtime:
            self._cache[path] = (mtime, pd.read_parquet(path))
        return self._cache[path][1]


def _mtime(path: Path) -> str | None:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None


def create_app(cfg: Config | None = None, client: FPLClient | None = None) -> FastAPI:
    cfg = cfg or load_config()
    client = client or make_client(cfg)
    files = _ParquetCache()
    app = FastAPI(title="FPL Breakout & Differential Finder", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def players() -> pd.DataFrame:
        preds = files.read(cfg.paths.predictions) if cfg.paths.predictions.exists() else None
        return merged_players(files.read(cfg.paths.live_players), preds)

    def fixtures_by_team(live: pd.DataFrame) -> dict[int, list[dict]]:
        teams = dict(zip(live["team_id"].astype(int), live["team_short"]))
        return upcoming_fixtures(client.fixtures(), teams, int(live["next_gw"].iloc[0]))

    def annotate(obj, fixtures: dict[str, list[dict]]):
        """Attach upcoming fixtures to every player dict in an optimizer result."""
        if isinstance(obj, dict):
            if {"id", "team", "position"} <= obj.keys():
                obj["fixtures"] = fixtures.get(obj["team"], [])
            for v in obj.values():
                annotate(v, fixtures)
        elif isinstance(obj, list):
            for v in obj:
                annotate(v, fixtures)
        return obj

    def by_short(live: pd.DataFrame) -> dict[str, list[dict]]:
        fx = fixtures_by_team(live)
        return {short: fx.get(int(tid), []) for tid, short in zip(live["team_id"], live["team_short"])}

    @app.exception_handler(FileNotFoundError)
    async def missing_data(request: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=503, content={
            "detail": f"Data not ready ({Path(str(exc)).name}). Run `python -m fplopt weekly` first."})

    @app.exception_handler(requests.HTTPError)
    async def fpl_error(request: Request, exc: requests.HTTPError):
        status = getattr(exc.response, "status_code", 502)
        if status == 404:
            return JSONResponse(status_code=404, content={"detail": "FPL has no team or league with that id."})
        return JSONResponse(status_code=502, content={"detail": f"The FPL API returned an error ({status})."})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        live = files.read(cfg.paths.live_players)
        gw = int(live["next_gw"].iloc[0])
        event = next((e for e in client.bootstrap()["events"] if e["id"] == gw), {})
        meta_file = cfg.paths.models / FINAL_MODEL / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        return {
            "season": live["season"].iloc[0], "gameweek": gw, "deadline": event.get("deadline_time"),
            "players": int(len(live)), "predictions_ready": cfg.paths.predictions.exists(),
            "data_updated": (client.cache_entry("bootstrap-static/") or {}).get("fetched_at"),
            "predictions_updated": _mtime(cfg.paths.predictions),
            "model_trained": meta.get("trained_at"), "model_train_seasons": meta.get("train_seasons"),
            "ownership_max_pct": cfg.differential.ownership_max_pct,
            "projections": sorted(SOURCES), "default_projection": cfg.optimizer.projection,
            "max_transfers": cfg.optimizer.max_transfers, "hit_cost": cfg.optimizer.hit_cost,
        }

    @app.get("/api/team/{team_id}")
    def team(team_id: int, projection: Projection | None = None,
             free_transfers: int | None = Query(None, ge=0, le=5), bank: float | None = Query(None, ge=0),
             max_transfers: int | None = Query(None, ge=0, le=4),
             differential_bonus: float | None = Query(None, ge=0, le=1)):
        result = optimize_team(cfg, client, team_id=team_id, free_transfers=free_transfers, bank=bank,
                               projection=projection, max_transfers=max_transfers,
                               differential_bonus=differential_bonus)
        return annotate(result, by_short(files.read(cfg.paths.live_players)))

    @app.get("/api/squad")
    def squad(projection: Projection | None = None, differential_bonus: float | None = Query(None, ge=0, le=1)):
        result = optimize_team(cfg, client, projection=projection, differential_bonus=differential_bonus)
        return annotate(result, by_short(files.read(cfg.paths.live_players)))

    @app.get("/api/differentials")
    def top_differentials(limit: int = Query(20, ge=1, le=100),
                          max_ownership: float | None = Query(None, gt=0, le=100),
                          position: Literal["GK", "DEF", "MID", "FWD"] | None = None,
                          max_price: float | None = Query(None, gt=0)):
        if not cfg.paths.predictions.exists():
            raise HTTPException(503, "No predictions yet. Run `python -m fplopt weekly` first.")
        live = players()
        own = max_ownership or cfg.differential.ownership_max_pct
        return {"gameweek": int(live["next_gw"].iloc[0]), "max_ownership": own,
                "players": differentials(live, fixtures_by_team(live), own, limit, position, max_price)}

    @app.get("/api/league/{league_id}")
    def league(league_id: int, team_id: int | None = None, sample: int = Query(20, ge=1, le=50)):
        live = players()
        gw = max(e["id"] for e in client.bootstrap()["events"] if e.get("finished"))
        return league_view(client, league_id, gw, live, fixtures_by_team(live), team_id=team_id, sample=sample,
                           max_league_ownership=cfg.differential.ownership_max_pct)

    return app
