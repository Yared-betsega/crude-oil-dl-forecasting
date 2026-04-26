import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

TICKER           = "CL=F"
START            = "2018-01-01"
LOOK_BACK        = 60
N_ANIMATION_DAYS = 30
TARGET_COL       = 3    # Close

DATA_DIR   = "data"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
DEV_DIR    = os.path.join(DATA_DIR, "dev")
TEST_DIR   = os.path.join(DATA_DIR, "test")

for d in [TRAIN_DIR, DEV_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)


def fetch_raw() -> pd.DataFrame:
    df = yf.download(TICKER, start=START, interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv("wti_crude_daily.csv")
    print(f"Downloaded {len(df)} rows → wti_crude_daily.csv")
    return df


def split_and_save(df: pd.DataFrame):
    test_data  = df.iloc[-N_ANIMATION_DAYS:]
    rest       = df.iloc[:-N_ANIMATION_DAYS]
    split      = int(len(rest) * 0.8)
    train_data = rest.iloc[:split]
    dev_data   = rest.iloc[split:]

    train_data.to_csv(os.path.join(TRAIN_DIR, "train.csv"))
    dev_data.to_csv(os.path.join(DEV_DIR,   "dev.csv"))
    test_data.to_csv(os.path.join(TEST_DIR,  "test.csv"))

    print(f"train : {len(train_data)} rows  → {TRAIN_DIR}/train.csv")
    print(f"dev   : {len(dev_data)}  rows  → {DEV_DIR}/dev.csv")
    print(f"test  : {len(test_data)} rows  → {TEST_DIR}/test.csv")


if __name__ == "__main__":
    df = fetch_raw()
    split_and_save(df)
