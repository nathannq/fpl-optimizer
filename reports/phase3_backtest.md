# Phase 3 - optimizer backtest

Each season is replayed week by week. Every gameweek the optimizer sees only one source's out-of-sample projections plus prices and ownership going into the gameweek. It makes transfers (1 free a week, banking up to 5, -4 per extra, the selling-price rule, up to 2 a week, and one more only if it adds 0.5+ projected points), then picks the XI, bench order and captain. It is scored on actual points with automatic substitutions and the vice-captain rule.

The squad is picked from scratch at GW2, the first gameweek with known prices. Not modelled: chips, price changes within a gameweek, historical injury flags.

## Season totals

| season | source | gameweeks | total_points | mean_per_gw | projected_per_gw | transfers | hits | hit_points | captain_points | autosubs |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024-25 | last3 | 37 | 1,817 | 49.108 | 84.360 | 68 | 32 | 128 | 574 | 22 |
| 2024-25 | model | 37 | 2,019 | 54.568 | 54.329 | 39 | 3 | 12 | 498 | 18 |
| 2025-26 | last3 | 37 | 1,763 | 47.649 | 82.360 | 64 | 28 | 112 | 342 | 35 |
| 2025-26 | model | 37 | 2,153 | 58.189 | 53.976 | 39 | 3 | 12 | 524 | 23 |

## Head to head (actual points, paired by gameweek)

| season | comparison | points_per_gw | ci_low | ci_high | season_total | gw_wins | gw_ties |
|---|---|---|---|---|---|---|---|
| 2024-25 | model - last3 | 5.459 | -1.866 | 12.405 | 202 | 0.595 | 0 |
| 2025-26 | model - last3 | 10.541 | 4.351 | 17.054 | 390 | 0.649 | 0.027 |

`ci_low`/`ci_high`: 95% interval for the mean difference per gameweek, from resampling gameweeks.

## Sources

- `model` - the LightGBM points model, trained only on earlier seasons (walk-forward).
- `last3` - last-3-fixture average x fixtures this gameweek.
- `ep_next` - FPL's own projection, from live snapshots taken before each deadline. Historical files only hold a post-gameweek xP, so this source appears once a season has snapshots.
