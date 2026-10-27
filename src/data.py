import os
import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

def load_series(series_id: str) -> pd.Series:
    df = pd.read_csv(URL.format(series_id))
    date_col = df.columns[0]                 # name differs between FRED versions
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series_id
    return s

def build_dataset(start: str = "1990-01-01") -> pd.DataFrame:
    df = pd.concat([load_series("REVOLSL"), load_series("UNRATE")], axis=1, sort=True)
    return df.loc[start:].asfreq("MS")        # one row per month

def check_quality(df: pd.DataFrame) -> list[str]:
    problems = []
    if df.isna().any().any():
        problems.append("missing values")
    if df.index.duplicated().any():
        problems.append("duplicate dates")
    if not df.index.is_monotonic_increasing:
        problems.append("dates out of order")
    if (df["REVOLSL"] <= 0).any():
        problems.append("non-positive balances")
    full = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    if len(full.difference(df.index)) > 0:
        problems.append("missing months")
    return problems

if __name__ == "__main__":
    df = build_dataset()
    print("Problems before cleaning:", check_quality(df))
    df = df.loc[:df.dropna().index.max()]   # trim months at the end where a series is missing
    filled = int(df.isna().sum().sum())
    df = df.interpolate(limit=2)            # fill short gaps in the middle
    print(f"Filled {filled} missing values by interpolation")
    assert check_quality(df) == [], "data still has problems"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/monthly.csv")
    print(f"Saved {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
