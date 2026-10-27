# Model card: Monthly revolving credit forecast

## Champion vs challenger decision rule (written 2026-10-27, before the backtest was run)

The challenger (LightGBM) replaces the champion (ETS) only if ALL of these hold:
1. Its average MAPE across all backtest windows is lower than the champion's.
2. It has the lower MAPE in at least 60 percent of windows.
3. It is not worse than the champion on average in the stress periods
   (origins 2007-01 to 2010-12 and 2019-06 to 2021-06).

The champion must also beat the naive baseline on average MAPE to be kept at all.
Otherwise the naive model is the benchmark we have failed to beat, and that is reported.
