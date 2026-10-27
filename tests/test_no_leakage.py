import pandas as pd
from src.models import make_features


def test_features_ignore_the_future():
    df = pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True).asfreq("MS")
    changed = df.copy()
    changed.iloc[-12:] = changed.iloc[-12:] * 10  # scramble the last 12 months
    before = make_features(df).iloc[:-12]
    after = make_features(changed).iloc[:-12]
    pd.testing.assert_frame_equal(before, after)
