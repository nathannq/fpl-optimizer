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


def cmd_refresh(cfg: Config, args: argparse.Namespace) -> None:
    args.force, args.no_live = False, False
    for step in (cmd_download_history, cmd_build_features, cmd_baselines):
        step(cfg, args)


COMMANDS = {
    "download-history": (cmd_download_history, "download historical seasons (vaastav) into data/raw"),
    "fetch-live": (cmd_fetch_live, "fetch and cache the current season from the FPL API"),
    "build-features": (cmd_build_features, "build data/processed/features.parquet"),
    "baselines": (cmd_baselines, "evaluate the baselines and write reports/phase1_baselines.md"),
    "refresh": (cmd_refresh, "download-history + build-features + baselines"),
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    COMMANDS[args.command][0](load_config(args.config), args)


if __name__ == "__main__":
    main()
