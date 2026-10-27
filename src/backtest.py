import numpy as np
import pandas as pd


def rmse(actual, pred) -> float:
    """Typical size of the miss, in the same units as the data."""
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mape(actual, pred) -> float:
    """Typical miss as a percentage of the actual value."""
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def rolling_backtest(df, models: dict, horizon=12, min_train=180, step=3) -> pd.DataFrame:
    """Expanding window. min_train=180 months is 15 years of history.
    At each origin, every model sees only data up to that date."""
    rows = []
    for end in range(min_train, len(df) - horizon + 1, step):
        train = df.iloc[:end]
        actual = df["REVOLSL"].iloc[end:end + horizon].to_numpy()
        for name, fn in models.items():
            pred = fn(train, horizon)
            rows.append({
                "origin": train.index[-1],
                "model": name,
                "rmse": rmse(actual, pred),
                "mape": mape(actual, pred),
            })
    return pd.DataFrame(rows)
