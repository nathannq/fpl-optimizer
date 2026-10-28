# FPL Breakout & Differential Finder

A Fantasy Premier League tool built around a machine-learning model that looks for **breakout and
differential players**: players owned by few managers who are likely to score well. A transfer
optimizer and a web UI sit on top of it.

> **Status:** Phases 1-3 of 5 are complete (data pipeline, baselines, model, transfer optimizer). The web
> UI and polish follow. See [Roadmap](#roadmap).

## What's here so far

- **One clean table.** It has one row per player per gameweek, covering 2022-23 to the live 2026-27
  season (about 116k rows). Each row holds the points actually scored and 56 features known
  **before** that gameweek.
- **A strict no-leakage rule, enforced by tests.** Each test randomly rewrites every result and price
  from gameweek N onwards, then checks that no feature for gameweek N or earlier changes.
- **Honest baselines**, written up in [`reports/phase1_baselines.md`](reports/phase1_baselines.md),
  including a finding about FPL's historical "xP" (below).
- **A LightGBM model** that predicts points, haul probability and 3-gameweek points. It is validated
  walk-forward by season. On seasons it never saw, its top-10 differentials returned 38% of the time
  against 27% for the baseline ([`reports/phase2_model.md`](reports/phase2_model.md)).
- **A transfer optimizer**: a mixed-integer program that recommends transfers, the XI, bench order and
  captain for your real team. Replaying two full seasons, the model's projections scored **2,019 and
  2,153 points** against **1,817 and 1,763** for the baseline's
  ([`reports/phase3_backtest.md`](reports/phase3_backtest.md)).

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
make refresh          # = download-history -> build-features -> baselines -> train -> predict
```

| Command | Does |
|---|---|
| `python -m fplopt download-history` | Fetch historical seasons into `data/raw/` |
| `python -m fplopt fetch-live` | Fetch the current season from the API (about 670 player histories, cached for 12h) |
| `python -m fplopt build-features [--no-live]` | Write `data/processed/features.parquet`, including rows for the next gameweek |
| `python -m fplopt baselines` | Write `reports/phase1_baselines.md` |
| `python -m fplopt train` | Walk-forward evaluation, final models in `models/`, write `reports/phase2_model.md` (about 40s) |
| `python -m fplopt predict [--top 10]` | Predict the next gameweek, save `data/processed/predictions.parquet`, list the top differentials |
| `python -m fplopt optimize --team-id <id>` | Transfers (0/1/2 compared), XI, bench, captain for your team; omit `--team-id` for a new squad |
| `python -m fplopt backtest` | Replay past seasons with each projection source, write `reports/phase3_backtest.md` (about 1 min) |
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

## The model

### How it works

- **Three targets:** points next gameweek (regression), a haul of 8+ points (classifier), and points
  over the next 3 gameweeks (regression).
- **Two model families:**
  - ridge regression, with L2 logistic regression for hauls (median imputation, missing-value
    indicators, scaled features, one-hot position);
  - LightGBM, which handles missing values and position natively.
- **Validation is walk-forward by season.** Each season is predicted only by models trained on earlier
  seasons. The number of boosting rounds comes from early stopping on the latest *training* season.
  Nothing is shuffled. 2024-25 is the **dev** season, used to choose the differential-score weights.
  2025-26 and 2026-27 to date are the **test** seasons, and are never used for any decision.
- **Reproducible.** All settings live in [`configs/config.yaml`](configs/config.yaml). LightGBM runs
  in deterministic mode with a fixed seed, so two runs produce a byte-identical report. Models are
  saved to `models/final/` with a `meta.json` recording the training seasons, tree counts, chosen
  weights, package versions and the data commit.

### The differential score

```
score = predicted points + haul_weight × P(haul) + edge_weight × (predicted − market-implied points) − price_weight × price
```

The score covers only players owned by fewer than 10% of managers going into the gameweek
(configurable). *Market-implied points* is what ownership, price, position and fixtures predict on
their own, i.e. "what ownership implies". The weights are picked from a grid by precision@10 on the
dev season.

The honest outcome: **every extra term made dev results worse.** Upweighting haul probability,
rewarding the edge over the market, and penalising price all lost to plain predicted points. Ranking
by edge alone did worse than the baseline on dev (0.27 against 0.32 precision), because it favours
fringe players the crowd is rightly ignoring. So the chosen weights are all 0, and the score is currently *predicted points among players
owned by less than 10%*. The other terms stay in the config, and the weights are re-chosen whenever the
model is retrained.

### Results on the test seasons (2025-26 and 2026-27, 44 gameweeks)

| | Last-3 average (baseline) | Ridge | LightGBM |
|---|---|---|---|
| Points MAE | 1.105 | 1.057 | **1.018** |
| Points Spearman (per GW) | 0.703 | 0.698 | **0.710** |
| Haul AUC | 0.825 | 0.855 | **0.860** |
| 3-GW points MAE | 2.73 | 2.40 | **2.34** |
| **Top-10 differentials: precision@10 (5+ pts)** | 0.268 | 0.357 | **0.382** |
| Top-10 differentials: haul rate (8+) | 0.141 | 0.200 | **0.214** |
| Top-10 differentials: mean points | 3.50 | 4.23 | **4.43** |

For comparison, an average differential scores 1.06 points and returns 7.6% of the time.

**Is the gain real?** In a gameweek-paired bootstrap against the baseline's top 10, LightGBM adds
**+11.4 percentage points of precision** (95% CI +5.7 to +17.3) and **+0.92 points per pick** (95% CI
+0.52 to +1.34). It wins 64% of gameweeks and ties 18%. The gain holds in every season: 2024-25,
2025-26 and 2026-27 so far.

**Against FPL's xP:** on gameweeks where it was recorded, the look-ahead xP still has a lower MAE
(0.87 against LightGBM's 1.02). That's expected, because it partly saw the result. A clean ep_next
comparison builds up from the live snapshots.

**Modest where it should be.** For predicting every player's points, the model only slightly beats
last-3 on ranking (Spearman 0.710 against 0.703). Most of that ranking is the easy part: who plays and
who doesn't. The real gain is in the tail, picking which low-owned players return, which is the job it
was built for.

### What drives it

Mean |SHAP| on the final model, using LightGBM's built-in TreeSHAP:

- **Points:** minutes in the last match by a distance, then points per match this season, minutes
  over the last 3, price and ownership.
- **Hauls:** points per match this season and the minutes features, then ownership and price.

Price and ownership carry real signal. The market is informative, which is also why "edge over the
market" makes a poor ranking on its own. Full tables are in
[`reports/phase2_model.md`](reports/phase2_model.md).

### Live predictions

`python -m fplopt predict` scores the next gameweek with the final model, which is trained on
everything including the current season to date. It then makes three adjustments:

- **Availability:** each prediction is multiplied by FPL's live chance of playing. Injured, suspended
  or departed players get 0.
- **Eligibility:** a player qualifies as a differential based on ownership *right now*.
- **Reasons:** each player gets three, taken from SHAP values and measured against a typical regular
  starter, e.g. *"xA per match, last 5 0.23 (+0.2)"*. Measuring against all players would put "plays
  90 minutes" first on every starter's list.

## The optimizer

`python -m fplopt optimize --team-id 1234567` (or `make optimize TEAM=1234567`) prints the recommended
transfers, the 0/1/2-transfer comparison, the starting XI by position, the captain and vice-captain,
and the bench order. Options:

- `--projection model|ep_next` swaps the projection source between this model and FPL's own.
- `--differential-bonus 0.1` adds +10% to differentials' projected points in the objective.
- `--free-transfers` and `--bank` override the inferred values.
- `--max-transfers N` compares more options.

### The formulation

A mixed-integer program (PuLP, solved by [HiGHS](https://highs.dev)) over every player. For each player
it decides *in squad*, *starting* and *captain*:

```
maximise   XI points + captain's points again + 0.1 × bench points − 4 × hits
subject to 2 GK / 5 DEF / 5 MID / 3 FWD; at most 3 per club
           XI of 11: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD; captain in the XI
           spend on new players ≤ bank + selling prices of players sold
           hits ≥ transfers − free transfers
           no buying or starting players flagged injured/suspended/left, or under 75% to play
```

It optimizes the **next gameweek only**. Each transfer count from 0 to 2 is solved exactly, which
takes about 0.1s per solve. One more transfer is recommended only if it adds at least 0.5 projected
points, so marginal moves don't use up a free transfer that could be banked. The bench is ordered as
FPL substitutes: reserve goalkeeper first, then outfielders by projection.

### Your real team, from public endpoints only

No login is needed. The public API doesn't expose selling prices, so they are rebuilt from your
transfer history:

- **Purchase price:** the cost of the latest transfer that brought each player in. Players from your
  original squad use their price in the gameweek you joined.
- **Selling price:** FPL's rule, under which you keep half of any rise (rounded down) and take every
  fall in full.
- **Free Hit weeks:** transfers in a Free Hit week are ignored and the squad reverts.
- **Pending transfers:** transfers you have already made for the coming gameweek are applied.
- **Free transfers:** replayed from your history. That means +1 a week, banking up to 5, with
  Wildcard and Free Hit weeks keeping saved transfers. Any of these can be overridden on the command
  line.

### Backtest: does a better projection win more points?

Each season is replayed week by week. Every gameweek, the optimizer sees only out-of-sample
projections plus prices and ownership going into the week. It makes transfers under the real rules,
then is scored on actual points with automatic substitutions and the vice-captain rule.

| Season | Projection | Points | Per GW | Projected per GW | Transfers | Hits |
|---|---|---|---|---|---|---|
| 2024-25 | last-3 average | 1,817 | 49.1 | 84.4 | 68 | 32 |
| 2024-25 | **model** | **2,019** | **54.6** | 54.3 | 39 | 3 |
| 2025-26 | last-3 average | 1,763 | 47.6 | 82.4 | 64 | 28 |
| 2025-26 | **model** | **2,153** | **58.2** | 54.0 | 39 | 3 |

The **model wins by +5.5 points per gameweek in 2024-25** (95% CI −1.9 to +12.4, so not conclusive on
its own) and by **+10.5 in 2025-26** (95% CI +4.4 to +17.1). It wins 60-65% of gameweeks.

The "projected per GW" column shows why. The last-3 average promised 82-84 points a week for the team
it picked and delivered 48-49. Optimizing over noisy projections selects whoever is on a lucky
streak, so the optimizer keeps chasing form and pays for it in hits (28-32 a season). The model's
projections for its own team were honest, at 54 projected against 55-58 delivered, so it took only 3
hits a season.

Simplifications: there are no chips; transfers use the previous gameweek's prices; there are no
historical injury flags; GW1 is skipped because prices are only known from the previous gameweek.

FPL's own `ep_next` is a supported source in both `optimize` and `backtest`. The backtest picks it up
automatically once the archived live snapshots cover a season. The historical files only have a
look-ahead xP, so the comparison starts with 2026-27.

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
  models/                  datasets & folds, ridge/LightGBM, differential score, SHAP reasons, report
  optimize/                rules, MILP solver, projection sources, team loader, backtest, rendering
  baselines.py, evaluate.py
  pipeline.py, cli.py      python -m fplopt <command>
tests/                     leakage, feature values, data, API cache, metrics, models, optimizer
reports/                   generated results (committed)
```

## Roadmap

1. ~~Data pipeline and baselines~~
2. ~~Model: ridge and LightGBM, walk-forward validation, differential score, evaluation against
   baselines~~
3. ~~Optimizer: PuLP MILP for next-gameweek transfers, with hits, the selling-price rule,
   availability and a season backtest~~
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
  no defensive-contribution features. The 2025-26 test fold was trained without any.
- **Availability is counted twice.** The live chance-of-playing multiplier partly overlaps with what
  the model already learned from minutes.
- **Next gameweek only.** The optimizer doesn't value rolling a transfer for a future double
  gameweek, beyond the 0.5-point threshold. A multi-gameweek planner is a designed-for extension.
- **Free-transfer inference.** One-off FPL top-ups are not modelled. Use `--free-transfers` if the
  inferred number is wrong.
- **The weight grid is small.** Precision@10 is noisy over a single dev season, so the differential
  weights are chosen from a deliberately small grid.
