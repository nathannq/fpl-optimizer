"""Command line entry point: ``python -m fplopt <command>`` (or ``fplopt <command>``)."""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from fplopt.config import Config, load_config


def cmd_download_history(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.data.vaastav import download_history

    manifest = download_history(cfg.history, cfg.paths.raw_history, force=args.force)
    print(f"{len(manifest['files'])} files from {manifest['repo']}@{manifest['sha'][:10]} "
          f"in {cfg.paths.raw_history}")


def cmd_fetch_live(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.data.live import load_live_season
    from fplopt.pipeline import make_client

    season, state = load_live_season(make_client(cfg), cfg.live.season)
    print(f"Season {state.season}: last finished GW{state.last_finished_gw}, next GW{state.next_gw} "
          f"(deadline {state.deadline}); {len(season.players)} players, {len(season.matches):,} match rows; "
          f"bootstrap fetched {state.bootstrap_fetched_at}")


def cmd_build_features(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.pipeline import build_features

    table = build_features(cfg, live=not args.no_live)
    summary = table.groupby("season").agg(gameweeks=("gw", "nunique"), rows=("gw", "size"),
                                          future_rows=("is_future", "sum"))
    print(summary.to_string())


def cmd_baselines(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.baselines import baseline_report

    report = baseline_report(pd.read_parquet(cfg.paths.features), cfg)
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.reports / "phase1_baselines.md"
    out.write_text(report)
    print(report)
    print(f"Saved {out}")


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.pipeline import run_training

    run_training(cfg)
    print(f"Saved models to {cfg.paths.models} and report to {cfg.paths.reports / 'phase2_model.md'}")


def cmd_predict(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.pipeline import predict_upcoming, top_differentials

    preds = predict_upcoming(cfg)
    gw = int(preds["gw"].iloc[0])
    cols = ["web_name", "team_short", "position", "now_cost", "ownership_now", "pred_points", "p_haul",
            "differential_score", "reasons"]
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        print(f"GW{gw} - top {args.top} differentials (owned by < {cfg.differential.ownership_max_pct:g}%)")
        print(top_differentials(preds, args.top)[cols].round(2).to_string())
    print(f"Saved {len(preds)} predictions to {cfg.paths.predictions}")


def cmd_optimize(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.optimize.render import render
    from fplopt.optimize.service import optimize_team
    from fplopt.pipeline import make_client

    result = optimize_team(cfg, make_client(cfg), team_id=args.team_id, free_transfers=args.free_transfers,
                           bank=args.bank, projection=args.projection, max_transfers=args.max_transfers,
                           differential_bonus=args.differential_bonus)
    print(render(result))


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.pipeline import run_backtest_report

    print(run_backtest_report(cfg))


def cmd_serve(cfg: Config, args: argparse.Namespace) -> None:
    import uvicorn

    from fplopt.web.app import create_app

    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning")


def cmd_weekly(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.pipeline import run_weekly

    summary, path = run_weekly(cfg, team_id=args.team_id, fresh=not args.cached, retrain=not args.no_retrain)
    print(summary)
    print(f"Saved {path}")


def cmd_charts(cfg: Config, args: argparse.Namespace) -> None:
    from fplopt.charts import make_charts

    for path in make_charts(cfg):
        print(f"Saved {path}")


COMMANDS = {
    "download-history": (cmd_download_history, "download historical seasons (vaastav) into data/raw"),
    "fetch-live": (cmd_fetch_live, "fetch and cache the current season from the FPL API"),
    "build-features": (cmd_build_features, "build data/processed/features.parquet"),
    "baselines": (cmd_baselines, "evaluate the baselines and write reports/phase1_baselines.md"),
    "train": (cmd_train, "walk-forward evaluation + final models -> models/, reports/phase2_model.md"),
    "predict": (cmd_predict, "predict the next gameweek and list the top differentials"),
    "optimize": (cmd_optimize, "recommend transfers, XI and captain (or a new squad without --team-id)"),
    "backtest": (cmd_backtest, "replay seasons with each projection source -> reports/phase3_backtest.md"),
    "serve": (cmd_serve, "run the web UI (FastAPI) on http://127.0.0.1:8000"),
    "charts": (cmd_charts, "draw the README result charts into docs/images/"),
    "weekly": (cmd_weekly, "the weekly run: fresh data, features, retrain, predict, your team's plan"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fplopt", description=__doc__)
    parser.add_argument("--config", help="path to config YAML (default configs/config.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        if name == "download-history":
            p.add_argument("--force", action="store_true", help="re-download files already present")
        if name == "build-features":
            p.add_argument("--no-live", action="store_true", help="historical seasons only (no API calls)")
        if name == "predict":
            p.add_argument("--top", type=int, default=10, help="how many differentials to list")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
        if name == "weekly":
            p.add_argument("--team-id", type=int, help="also plan transfers for this FPL team")
            p.add_argument("--no-retrain", action="store_true", help="reuse the saved model (faster)")
            p.add_argument("--cached", action="store_true", help="allow cached API data instead of refetching")
        if name == "optimize":
            p.add_argument("--team-id", type=int, help="your FPL team id (omit to build a new squad)")
            p.add_argument("--projection", choices=["model", "ep_next"], help="projection source (default: config)")
            p.add_argument("--free-transfers", type=int, help="override the inferred free transfers")
            p.add_argument("--bank", type=float, help="override the bank, in £m")
            p.add_argument("--max-transfers", type=int, help="compare 0..N transfers (default: config)")
            p.add_argument("--differential-bonus", type=float,
                           help="e.g. 0.1 = +10%% weight on differentials' projected points")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    COMMANDS[args.command][0](load_config(args.config), args)


if __name__ == "__main__":
    main()
