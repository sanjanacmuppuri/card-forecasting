import pandas as pd
from src.data import check_quality


def test_clean_data_passes():
    df = pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True)
    assert check_quality(df) == []
