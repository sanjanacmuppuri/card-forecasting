# Model card: Monthly revolving credit forecast

## Champion vs challenger decision rule (written 2026-10-27, before the backtest was run)

The challenger (LightGBM) replaces the champion (ETS) only if ALL of these hold:
1. Its average MAPE across all backtest windows is lower than the champion's.
2. It has the lower MAPE in at least 60 percent of windows.
3. It is not worse than the champion on average in the stress periods
   (origins 2007-01 to 2010-12 and 2019-06 to 2021-06).

The champion must also beat the naive baseline on average MAPE to be kept at all.
Otherwise the naive model is the benchmark we have failed to beat, and that is reported.

## Backtest results and decision

Windows: 83  (first origin 2004-12-01, last 2025-06-01)

Average scorecard (all windows):
                     rmse  mape
model                          
challenger_lgbm  35656.73  2.90
champion_ets     26936.16  2.25
naive            36606.74  3.14 

Challenger beats champion in 37% of windows

Stress periods only:
                     rmse  mape
model                          
challenger_lgbm  52708.90  4.50
champion_ets     44743.38  3.82
naive            45533.80  4.10

## Backtest results and decision

Design: expanding window, 12-month horizon, 83 windows (origins 2004-12 to 2025-06), step of 3 months.

| Model | RMSE | MAPE (all windows) | MAPE (stress periods) |
| --- | --- | --- | --- |
| Naive | 36,607 | 3.14% | 4.10% |
| Champion (ETS) | 26,936 | 2.25% | 3.82% |
| Challenger (LightGBM) | 35,657 | 2.90% | 4.50% |

Challenger beat the champion in 37% of windows.

Decision: keep the champion. The challenger failed all three conditions in the
pre-registered rule (average MAPE, 60% win share, stress periods). The champion
also beat the naive baseline, so it stays as the model of record.

Caveats: windows overlap (3-month step, 12-month horizon), so they are not independent
tests and the win share should be read roughly. The challenger was run with default
settings and not tuned, by design.
