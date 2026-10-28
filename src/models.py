import numpy as np
import pandas as pd
import lightgbm as lgb

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Baseline: repeat the last observed balance."""
    return np.repeat(train["REVOLSL"].iloc[-1], horizon)


def ets_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Champion: damped-trend exponential smoothing on log balances."""
    log_y = np.log(train["REVOLSL"])
    fit = ExponentialSmoothing(log_y, trend="add", damped_trend=True).fit()
    return np.exp(fit.forecast(horizon).to_numpy())

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature uses only information known at that month."""
    g = np.log(df["REVOLSL"]).diff()
    X = pd.DataFrame(index=df.index)
    for k in (0, 1, 2, 5, 11):  # growth now, and 1, 2, 5, 11 months ago
        X[f"growth_lag{k}"] = g.shift(k)
    X["growth_avg3"] = g.rolling(3).mean()
    X["growth_avg12"] = g.rolling(12).mean()
    X["unrate"] = df["UNRATE"]
    X["unrate_chg3"] = df["UNRATE"].diff(3)
    X["month"] = df.index.month
    return X

def lgbm_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Challenger: one gradient boosting model per forecast month."""
    X = make_features(train)
    log_y = np.log(train["REVOLSL"])
    last_x, last_level = X.iloc[[-1]], log_y.iloc[-1]
    out = []
    for h in range(1, horizon + 1):
        target = (log_y.shift(-h) - log_y).rename("target")  # growth over next h months
        data = X.join(target).dropna()
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=8,
            min_child_samples=20, random_state=0, verbose=-1,
        )
        model.fit(data.drop(columns="target"), data["target"])
        out.append(np.exp(last_level + model.predict(last_x)[0]))
    return np.array(out)
