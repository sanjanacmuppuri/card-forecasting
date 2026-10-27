import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Baseline: repeat the last observed balance."""
    return np.repeat(train["REVOLSL"].iloc[-1], horizon)


def ets_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Champion: damped-trend exponential smoothing on log balances."""
    log_y = np.log(train["REVOLSL"])
    fit = ExponentialSmoothing(log_y, trend="add", damped_trend=True).fit()
    return np.exp(fit.forecast(horizon).to_numpy())
