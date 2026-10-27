import numpy as np


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index: how much a distribution has shifted."""
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def needs_review(recent_mape, backtest_mape, psi_value) -> bool:
    """Your own documented thresholds, not industry law."""
    return recent_mape > 1.5 * backtest_mape or psi_value > 0.25
