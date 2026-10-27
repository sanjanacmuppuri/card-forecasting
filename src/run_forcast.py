name: monthly-refresh
on:
  schedule:
    - cron: "0 12 8 * *"      # 8th of every month
  workflow_dispatch:
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest
      - run: python -m src.run_forecast

