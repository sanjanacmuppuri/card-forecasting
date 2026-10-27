import pandas as pd
import streamlit as st
import lightgbm as lgb

from src.data import build_dataset
from src.models import ets_forecast, lgbm_forecast

st.title("Revolving credit forecast")

df = build_dataset().dropna().asfreq("MS")
h = st.slider("Months ahead", 1, 12, 6)

idx = pd.date_range(df.index[-1], periods=h + 1, freq="MS")[1:]
forecasts = pd.DataFrame(
    {"champion": ets_forecast(df, h), "challenger": lgbm_forecast(df, h)}, index=idx
)
history = df["REVOLSL"].iloc[-60:].rename("actual")
st.line_chart(pd.concat([history, forecasts], axis=1))

st.caption("Personal project using public Federal Reserve data via FRED. Not investment advice.")
