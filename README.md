# FPL Breakout & Differential Finder

A Fantasy Premier League tool built around a machine-learning model that looks for **breakout and
differential players**: players owned by few managers who are likely to score well. A transfer
optimizer and a web UI sit on top of it.

> **Status:** Phase 1 of 5 is complete (data pipeline and baselines). The model, optimizer, web UI and
> polish follow. See [Roadmap](#roadmap).

## What's here so far

- **One clean table.** It has one row per player per gameweek, covering 2022-23 to the live 2026-27
  season (about 116k rows). Each row holds the points actually scored and 56 features known
  **before** that gameweek.
- **A strict no-leakage rule, enforced by tests.** Each test randomly rewrites every result and price
  from gameweek N onwards, then checks that no feature for gameweek N or earlier changes.
- **Honest baselines**, written up in [`reports/phase1_baselines.md`](reports/phase1_baselines.md),
  including a finding about FPL's historical "xP" (below).

## Setup

Requires Python 3.11+. On macOS, LightGBM also needs OpenMP (`brew install libomp`).

```bash
python3 -m venv .venv            # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt -e .
```

`make setup` does the same thing.

## Data

| Source | What | How |
|---|---|---|
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | Completed seasons 2022-23 to 2025-26: per-fixture player stats, prices, ownership | `make data` |
| Official FPL API | The current season, upcoming fixtures, your team and mini-leagues | `make live` |

**Historical data.** `python -m fplopt download-history` downloads four files per season
(`gws/merged_gw.csv`, `players_raw.csv`, `teams.csv`, `fixtures.csv`) into `data/raw/vaastav/` from
a commit pinned in [`configs/config.yaml`](configs/config.yaml). It writes `MANIFEST.json` with the
commit SHA and each file's SHA-256. Re-running skips files that are already up to date. Set
`history.ref: master` to take the latest data instead. Everything under `data/` is gitignored.

**Live data.** The API client ([`src/fplopt/data/api.py`](src/fplopt/data/api.py)) caches every
response in `data/cache/api/` as `{fetched_at, url, data}`. A response is served from a fresh cache
entry if one exists, otherwise from the network. If the network fails, the client falls back to a
stale entry and logs a warning. Each `bootstrap-static` fetch is also archived to
`data/snapshots/`, because FPL's `ep_next` exists nowhere else.

Seasons 2022-23 onward are used because they are the first with xG/xA.

## The pipeline

```bash
make refresh          # = download-history -> build-features -> baselines
```

| Command | Does |
|---|---|
| `python -m fplopt download-history` | Fetch historical seasons into `data/raw/` |
| `python -m fplopt fetch-live` | Fetch the current season from the API (about 670 player histories, cached for 12h) |
| `python -m fplopt build-features [--no-live]` | Write `data/processed/features.parquet`, including rows for the next gameweek |
| `python -m fplopt baselines` | Write `reports/phase1_baselines.md` |
| `make test` | Run the test suite (no network needed) |

### Features (all known before the gameweek)

| Group | Features |
|---|---|
| Player form | 3- and 5-fixture averages of points, minutes, starts, goals, assists, xG, xA, xGI, bonus, BPS, clean sheets, defensive contributions; xG, xA and points per 90 |
| Availability / rotation | minutes last match, minutes trend (last 3 vs previous 3), matches since last start, played last match |
| Market | price, price change (last GW and season), ownership %, ownership change, net transfers % |
| Team | rolling xG for/against, goals for/against, points per game, clean-sheet rate (last 6 matches) |
| Fixture | number of fixtures this gameweek (doubles/blanks), home share, opponent attack/defence strength, opponent strength over the next 3 gameweeks |
| Other | position, points per match this season, matches so far this season |

Targets are points in the gameweek, haul (8+ points) and points over the next 3 gameweeks.

### How leakage is prevented

Each feature is computed as a "state after gameweek G", carried over gameweeks the player didn't
play, then **shifted by one gameweek**. As a result, a gameweek-N row only ever sees results up to
N−1. The table follows these rules:

- **Price, ownership and transfers** come from the previous gameweek's record. Using the current
  gameweek's figures would leak, because they are recorded at the deadline, after you would have
  made your decision.
- **Team and opponent strength** are calculated from results we already have, **not from FPL's
  fixture difficulty ratings**. The historical FDR files are end-of-season snapshots.
- **Ownership %** is ownership divided by an estimated number of managers. Every squad holds 15
  players, so total ownership ÷ 15 = managers. This estimate never uses a later gameweek.

`tests/test_leakage.py` enforces the rule on synthetic data and, when it has been downloaded, on
real data. Deliberately injected leaks fail 14 of its 16 tests.

## Baselines and the FPL xP problem

These numbers cover the 80 gameweeks in 2023-24 to 2025-26 where FPL xP was recorded. A
differential is owned by fewer than 10% of managers, and a return is 5+ points.

| Projection | MAE | Spearman | Top-10 differentials: return rate | Mean pts |
|---|---|---|---|---|
| Last-3 average (honest baseline) | 1.07 | 0.69 | 30% | 3.4 |
| FPL xP as recorded (look-ahead) | 0.85 | 0.75 | 69% | 7.6 |
| Average differential (no model) | | | 7% | 1.0 |

The data records FPL's xP **after** each gameweek, and by then FPL's "form" input already includes
that gameweek's points. The clearest evidence: among players with no minutes in their previous 5
fixtures, recorded xP averages 2.6 for those who went on to haul and 0.35 for those who blanked. A
real forecast can't tell those groups apart in advance. Its "top-10 differentials" hit rate of 69%
is not achievable by any honest model. Because of this:

- The **last-3 average** is the honest bar to beat.
- The **recorded xP** is reported only as an optimistic upper reference.
- A **clean `ep_next` comparison** builds up from now on, using the live snapshots.

Full tables: [`reports/phase1_baselines.md`](reports/phase1_baselines.md).

## Data quirks handled

- **Missing xP.** xP was not captured in some gameweeks (all zeros), including most of 2025-26.
  Those gameweeks are masked.
- **Late xG and starts.** FPL only added xG/xA and starts in 2022-23 gameweek 16. Before that they
  read 0 for every player and are treated as missing.
- **Assistant managers.** The 2024-25 "assistant manager" pseudo-players are dropped.
- **Duplicate rows.** 2025-26 has duplicate player-fixture rows, which are dropped.
- **Postponed gameweek.** 2022-23 has no gameweek 7, which was postponed.
- **Double and blank gameweeks.** Double gameweeks are summed into one row. Blanks have no row.
- **Changing player ids.** Players are linked across seasons by FPL's stable `code`, because the
  per-season `id` changes.

## Project layout

```
configs/config.yaml        all settings (seasons, windows, thresholds)
src/fplopt/
  data/                    API client, vaastav downloader, live season, common schema
  features/                stacked tables and the leak-free feature builder
  baselines.py, evaluate.py
  cli.py                   python -m fplopt <command>
tests/                     leakage, feature values, data, API cache, metrics
reports/                   generated results (committed)
```

## Roadmap

1. ~~Data pipeline and baselines~~
2. Model: ridge, then LightGBM, for points, haul probability and 3-gameweek points; time-based
   validation; differential score; precision@10 against baselines; feature importance
3. Optimizer: PuLP MILP for next-gameweek transfers, with hits, the selling-price rule and
   availability
4. Web UI: FastAPI and plain HTML/JS, with a pitch view, differentials page and mini-league view
5. Polish: README with results chart and screenshots, a weekly one-command refresh

Designed for, but not yet built: chip planning and a multi-gameweek planner.

## Limitations (so far)

- **No historical injury flags.** FPL's `status` and `chance_of_playing` have no history, so the
  model learns availability from minutes patterns. Live flags are applied when serving.
- **Final fixture schedule.** The historical schedule is the final one. A double gameweek two weeks
  ahead may not have been announced at the time, which slightly flatters the 3-gameweek fixture
  features.
- **Defensive contributions.** These have scored points only since 2025-26, so older seasons have
  no defensive-contribution features.
