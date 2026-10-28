# Phase 2 - the model

## Setup

- **Walk-forward by season.** Each season is predicted by models trained only on earlier seasons (dev: 2024-25; test: 2025-26, 2026-27). The number of boosting rounds is chosen by early stopping on the latest *training* season. Nothing is shuffled.
- **Dev vs test.** The differential-score weights were chosen on the dev season only. The test seasons were not used for any decision.
- **Targets:** points next GW (regression), haul = 8+ points (classifier), points over the next 3 GWs (regression).
- **Models:** ridge / L2-logistic (median imputation + missing indicators, scaled) and LightGBM (seed 42, deterministic). Final-model tree counts: lgbm_points 212, lgbm_haul 231, lgbm_next3 321.
- **Differentials:** owned by < 10% going into the GW; top 10 per GW; a return is 5+ points.

## Points next gameweek

| split | seasons | projection | n | mae | rmse | bias | spearman |
|---|---|---|---|---|---|---|---|
| dev | 2024-25 | last-3 average | 26,919 | 1.076 | 2.171 | -0.000 | 0.692 |
| dev | 2024-25 | ridge | 26,919 | 1.063 | 1.944 | 0.010 | 0.691 |
| dev | 2024-25 | LightGBM | 26,919 | 1.018 | 1.928 | 0.021 | 0.708 |
| test | 2025-26, 2026-27 | last-3 average | 34,558 | 1.105 | 2.231 | -0.004 | 0.703 |
| test | 2025-26, 2026-27 | ridge | 34,558 | 1.057 | 1.982 | -0.055 | 0.698 |
| test | 2025-26, 2026-27 | LightGBM | 34,558 | 1.018 | 1.969 | -0.057 | 0.710 |

Same rows, restricted to gameweeks where FPL's xP was recorded. That xP was captured *after* each gameweek (see Phase 1), so it is an optimistic reference, not a fair competitor:

| projection | gameweeks | n | mae | rmse | bias | spearman |
|---|---|---|---|---|---|---|
| last-3 average | 46 | 33,076 | 1.089 | 2.192 | -0.002 | 0.690 |
| LightGBM | 46 | 33,076 | 1.021 | 1.937 | -0.000 | 0.706 |
| FPL xP as recorded (look-ahead) | 46 | 33,076 | 0.869 | 1.624 | -0.026 | 0.753 |

A fair comparison with FPL's `ep_next` needs projections captured before the deadline. The live pipeline archives them from 2026-27 GW9 onwards (`data/snapshots/`), so that comparison will grow week by week.

Calibration of the LightGBM points model on the test seasons:

| predicted points | rows | mean_predicted | mean_actual |
|---|---|---|---|
| < 0.5 | 17651 | 0.128 | 0.095 |
| 0.5-1 | 3131 | 0.736 | 0.792 |
| 1-2 | 4359 | 1.517 | 1.660 |
| 2-3 | 5226 | 2.498 | 2.715 |
| 3-4 | 3287 | 3.401 | 3.591 |
| 4-5 | 665 | 4.379 | 4.465 |
| 5-6 | 139 | 5.375 | 5.230 |
| 6+ | 100 | 7.199 | 6.580 |

## Haul probability (8+ points)

| split | model | auc | brier | log_loss | haul_rate |
|---|---|---|---|---|---|
| dev | base rate | 0.500 | 0.034 | 0.151 | 0.035 |
| dev | last-3 average as a score | 0.830 |  |  | 0.035 |
| dev | logistic (ridge) | 0.861 | 0.031 | 0.118 | 0.035 |
| dev | LightGBM | 0.868 | 0.031 | 0.117 | 0.035 |
| test | base rate | 0.500 | 0.037 | 0.165 | 0.039 |
| test | last-3 average as a score | 0.825 |  |  | 0.039 |
| test | logistic (ridge) | 0.855 | 0.034 | 0.130 | 0.039 |
| test | LightGBM | 0.860 | 0.034 | 0.129 | 0.039 |

## Points over the next 3 gameweeks

| split | projection | n | mae | rmse | bias | spearman |
|---|---|---|---|---|---|---|
| dev | last-3 average x fixtures | 25,354 | 2.646 | 4.921 | 0.012 | 0.745 |
| dev | ridge | 25,354 | 2.370 | 3.878 | 0.012 | 0.762 |
| dev | LightGBM | 25,354 | 2.324 | 3.841 | 0.030 | 0.779 |
| test | last-3 average x fixtures | 31,540 | 2.726 | 5.064 | 0.001 | 0.755 |
| test | ridge | 31,540 | 2.397 | 3.982 | -0.168 | 0.766 |
| test | LightGBM | 31,540 | 2.338 | 3.971 | -0.209 | 0.779 |

## Differential score

score = predicted points + haul_weight x P(haul) + edge_weight x (predicted - market-implied points) - price_weight x price, for players owned by < 10%.

Chosen on the dev season: **haul_weight = 0, edge_weight = 0, price_weight = 0**. Best weight combinations on dev:

| haul_weight | edge_weight | price_weight | precision | haul_rate | mean_points |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.381 | 0.214 | 4.719 |
| 2 | 0 | 0 | 0.365 | 0.222 | 4.668 |
| 5 | 0 | 0 | 0.359 | 0.211 | 4.549 |
| 10 | 0 | 0.250 | 0.357 | 0.216 | 4.519 |
| 10 | 0 | 0 | 0.357 | 0.211 | 4.505 |
| 10 | 1 | 0 | 0.357 | 0.216 | 4.478 |
| 2 | 1 | 0 | 0.351 | 0.203 | 4.327 |
| 5 | 1 | 0 | 0.346 | 0.197 | 4.297 |

## Differentials: top 10 per gameweek

| split | ranked by | gameweeks | precision@10 | haul_rate | mean_points |
|---|---|---|---|---|---|
| dev | last-3 average (baseline) | 37 | 0.324 | 0.130 | 3.757 |
| dev | market-implied (ownership + price) | 37 | 0.300 | 0.168 | 3.630 |
| dev | ridge: points | 37 | 0.378 | 0.208 | 4.659 |
| dev | LightGBM: points | 37 | 0.381 | 0.214 | 4.719 |
| dev | LightGBM: P(haul) | 37 | 0.357 | 0.232 | 4.622 |
| dev | LightGBM: edge over market | 37 | 0.273 | 0.132 | 3.646 |
| dev | differential score | 37 | 0.381 | 0.214 | 4.719 |
| test | last-3 average (baseline) | 44 | 0.268 | 0.141 | 3.502 |
| test | market-implied (ownership + price) | 44 | 0.273 | 0.130 | 3.300 |
| test | ridge: points | 44 | 0.357 | 0.200 | 4.232 |
| test | LightGBM: points | 44 | 0.382 | 0.214 | 4.425 |
| test | LightGBM: P(haul) | 44 | 0.327 | 0.168 | 3.948 |
| test | LightGBM: edge over market | 44 | 0.291 | 0.136 | 3.702 |
| test | differential score | 44 | 0.382 | 0.214 | 4.425 |

For reference, an average differential (no model) in the same pool:

| split | mean_points | return_rate |
|---|---|---|
| dev | 1.024 | 0.073 |
| test | 1.063 | 0.076 |

Season by season:

| season | ranked by | gameweeks | precision@10 | haul_rate | mean_points |
|---|---|---|---|---|---|
| 2024-25 | last-3 average (baseline) | 37 | 0.324 | 0.130 | 3.757 |
| 2024-25 | LightGBM: points | 37 | 0.381 | 0.214 | 4.719 |
| 2025-26 | last-3 average (baseline) | 37 | 0.270 | 0.141 | 3.519 |
| 2025-26 | LightGBM: points | 37 | 0.384 | 0.216 | 4.438 |
| 2026-27 | last-3 average (baseline) | 7 | 0.257 | 0.143 | 3.414 |
| 2026-27 | LightGBM: points | 7 | 0.371 | 0.200 | 4.357 |

### Is it better than the baseline?

Paired by gameweek against the last-3 average's top 10. `diff` is the mean per-gameweek difference; the 95% CI comes from resampling gameweeks. `gw_wins` and `gw_ties` are the shares of gameweeks that were better or equal.

| split | ranked by | metric | diff | ci_low | ci_high | gw_wins | gw_ties |
|---|---|---|---|---|---|---|---|
| dev | LightGBM: points | precision | 0.057 | 0.008 | 0.108 | 0.514 | 0.243 |
| dev | LightGBM: points | mean_points | 0.962 | 0.603 | 1.341 | 0.811 | 0 |
| dev | LightGBM: P(haul) | precision | 0.032 | -0.024 | 0.095 | 0.405 | 0.324 |
| dev | LightGBM: P(haul) | mean_points | 0.865 | 0.438 | 1.324 | 0.757 | 0.027 |
| dev | differential score | precision | 0.057 | 0.008 | 0.108 | 0.514 | 0.243 |
| dev | differential score | mean_points | 0.962 | 0.603 | 1.341 | 0.811 | 0 |
| test | LightGBM: points | precision | 0.114 | 0.057 | 0.173 | 0.636 | 0.182 |
| test | LightGBM: points | mean_points | 0.923 | 0.520 | 1.339 | 0.773 | 0.045 |
| test | LightGBM: P(haul) | precision | 0.059 | 0.005 | 0.116 | 0.523 | 0.182 |
| test | LightGBM: P(haul) | mean_points | 0.445 | 0.082 | 0.820 | 0.614 | 0.023 |
| test | differential score | precision | 0.114 | 0.057 | 0.173 | 0.636 | 0.182 |
| test | differential score | mean_points | 0.923 | 0.520 | 1.339 | 0.773 | 0.045 |

## What drives the predictions

Mean absolute SHAP value (LightGBM's TreeSHAP) on the final model, over the last two seasons. Points model in points; haul model in log-odds.

### Points model

| feature | description | mean_abs_shap |
|---|---|---|
| mins_last | minutes last match | 0.680 |
| pts_season_avg | points per match this season | 0.172 |
| mins_r3 | minutes per match, last 3 | 0.171 |
| played_last | played last match | 0.118 |
| price | price | 0.082 |
| ownership_pct | ownership | 0.068 |
| home_frac | at home | 0.052 |
| n_fixtures | fixtures this GW | 0.036 |
| mins_r5 | minutes per match, last 5 | 0.033 |
| xgi_r3 | xGI per match, last 3 | 0.026 |
| defcon_r3 | defensive contributions per match, last 3 | 0.025 |
| opp_xg_for | opponent xG per match | 0.025 |
| opp_goals_for | opponent goals per match | 0.024 |
| pts_r3 | points per match, last 3 | 0.023 |
| opp_xg_against | opponent xG conceded per match | 0.023 |

### Haul model

| feature | description | mean_abs_shap |
|---|---|---|
| pts_season_avg | points per match this season | 0.536 |
| mins_r3 | minutes per match, last 3 | 0.475 |
| mins_r5 | minutes per match, last 5 | 0.411 |
| mins_last | minutes last match | 0.402 |
| ownership_pct | ownership | 0.147 |
| price | price | 0.100 |
| pts_r3 | points per match, last 3 | 0.087 |
| matches_since_start | matches since last start | 0.065 |
| ownership_change_1gw | ownership change last GW | 0.058 |
| opp_goals_for | opponent goals per match | 0.056 |
| home_frac | at home | 0.055 |
| opp_xg_for | opponent xG per match | 0.055 |
| net_transfers_pct | net transfers last GW | 0.047 |
| opp_xg_against | opponent xG conceded per match | 0.039 |
| mins_trend | minutes trend | 0.037 |
