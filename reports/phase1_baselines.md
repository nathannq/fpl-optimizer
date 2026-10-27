# Phase 1 - baselines

Evaluation seasons: 2023-24, 2024-25, 2025-26, 2026-27. A differential is owned by < 10% of managers going into the gameweek; a return is >= 5 points and a haul >= 8.

## Data coverage

| season | gameweeks | rows | gameweeks_with_fpl_xp |
|---|---|---|---|
| 2023-24 | 38 | 28742 | 37 |
| 2024-25 | 38 | 26919 | 35 |
| 2025-26 | 38 | 29338 | 11 |
| 2026-27 | 8 | 5220 | 0 |

## Is FPL's recorded xP a fair baseline?

Players with **zero minutes in their previous 5 fixtures**, grouped by what they went on to score:

| outcome in GW | rows | mean FPL xP |
|---|---|---|
| hauled (8+ pts) | 30 | 2.640 |
| returned (3-7 pts) | 122 | 1.526 |
| played, blanked (1-2 pts) | 819 | 0.351 |

A projection made before the deadline cannot know which idle players will haul. The recorded xP does, because the source captured it after the gameweek (FPL's form input already includes that gameweek). `fpl_xp` is therefore reported as an optimistic, look-ahead reference; `last3` is the honest baseline.

## Accuracy (points next gameweek)

| rows | baseline | n | mae | rmse | bias | spearman |
|---|---|---|---|---|---|---|
| all players | last3 | 90,219 | 1.075 | 2.201 | -0.003 | 0.692 |
| played in last 5 | last3 | 49,214 | 1.903 | 2.932 | 0.061 | 0.455 |
| all players, GWs with FPL xP | last3 | 61,169 | 1.065 | 2.191 | -0.002 | 0.686 |
| all players, GWs with FPL xP | fpl_xp | 61,169 | 0.847 | 1.614 | -0.032 | 0.747 |
| played in last 5, GWs with FPL xP | last3 | 33,623 | 1.868 | 2.907 | 0.065 | 0.457 |
| played in last 5, GWs with FPL xP | fpl_xp | 33,623 | 1.348 | 2.107 | -0.044 | 0.708 |

## Differentials: top 10 per gameweek

| season | baseline | gameweeks | precision@10 | haul_rate | mean_points | pool_mean_points | pool_return_rate |
|---|---|---|---|---|---|---|---|
| 2023-24 | last3 | 37 | 0.319 | 0.114 | 3.565 | 0.949 | 0.070 |
| 2024-25 | last3 | 37 | 0.324 | 0.130 | 3.757 | 1.024 | 0.073 |
| 2025-26 | last3 | 37 | 0.270 | 0.141 | 3.519 | 1.037 | 0.074 |
| 2026-27 | last3 | 7 | 0.257 | 0.143 | 3.414 | 1.227 | 0.087 |
| all | last3 | 118 | 0.302 | 0.129 | 3.602 | 1.014 | 0.073 |
| GWs with FPL xP | last3 | 80 | 0.300 | 0.106 | 3.428 | 0.991 | 0.072 |
| GWs with FPL xP | fpl_xp | 80 | 0.694 | 0.470 | 7.607 | 0.991 | 0.072 |

`pool_*` = the average differential that gameweek (what ownership alone implies).
