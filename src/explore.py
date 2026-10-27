import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save images instead of opening windows
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

os.makedirs("docs/figures", exist_ok=True)

df = pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True).asfreq("MS")
y = df["REVOLSL"]

RECESSIONS = [("2007-12-01", "2009-06-01"), ("2020-02-01", "2020-04-01")]

# 1. The level, with recessions shaded
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y.index, y.values)
for start, end in RECESSIONS:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2)
ax.set_title("Revolving consumer credit, $ millions")
fig.tight_layout()
fig.savefig("docs/figures/01_level.png", dpi=150)
plt.close(fig)

# 2. Monthly growth, which is what we will actually model
growth = np.log(y).diff()
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(growth.index, growth.values)
for start, end in RECESSIONS:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2)
ax.set_title("Monthly log growth")
fig.tight_layout()
fig.savefig("docs/figures/02_growth.png", dpi=150)
plt.close(fig)

# 3. Split into trend, seasonal, and leftover
result = STL(np.log(y), period=12).fit()
fig = result.plot()
fig.set_size_inches(10, 8)
fig.savefig("docs/figures/03_stl.png", dpi=150)
plt.close(fig)

# Numbers to describe in your notes
print("Overall monthly growth:")
print(growth.describe().round(4))
print("\nGreat Recession (2007-12 to 2009-06):")
print(growth["2007-12-01":"2009-06-01"].describe().round(4))
print("\nCOVID window (2020-02 to 2020-04):")
print(growth["2020-02-01":"2020-04-01"].round(4))
print("\nLargest drops in growth:")
print(growth.nsmallest(5).round(4))
