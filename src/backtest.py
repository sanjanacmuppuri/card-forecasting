import numpy as np


def rmse(actual, pred) -> float:
    """Typical size of the miss, in the same units as the data."""
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mape(actual, pred) -> float:
    """Typical miss as a percentage of the actual value."""
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)
