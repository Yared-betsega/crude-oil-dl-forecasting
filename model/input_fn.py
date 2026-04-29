import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

LOOK_BACK        = 10
TARGET_COL       = 3    # index of Close in [Open, High, Low, Close, Volume]
N_ANIMATION_DAYS = 30
BATCH_SIZE       = 64


def load_and_split(csv_path: str):
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

    train_test_data = data.iloc[:-N_ANIMATION_DAYS]
    anim_data_full  = data.iloc[-(N_ANIMATION_DAYS + LOOK_BACK):]
    return train_test_data, anim_data_full, data


def fit_scaler(train_test_data: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_test_data.values.astype(np.float32))
    return scaler


def create_sequences(arr: np.ndarray, look_back: int, target_col: int):
    X, y = [], []
    for i in range(len(arr) - look_back):
        X.append(arr[i : i + look_back])
        y.append(arr[i + look_back, target_col])
    return np.array(X), np.array(y)


def get_data_loaders(csv_path: str = "wti_crude_daily.csv"):
    train_test_data, anim_data_full, full_data = load_and_split(csv_path)

    scaler       = fit_scaler(train_test_data)
    scaled       = scaler.transform(train_test_data.values).astype(np.float32)
    scaled_anim  = scaler.transform(anim_data_full.values).astype(np.float32)

    X, y = create_sequences(scaled, LOOK_BACK, TARGET_COL)

    split         = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train)
    X_test_t  = torch.tensor(X_test)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    y_test_t  = torch.tensor(y_test).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    return {
        "train_loader": train_loader,
        "X_train":      X_train_t,
        "X_test":       X_test_t,
        "y_train":      y_train_t,
        "y_test":       y_test_t,
        "X_train_raw":  X_train,
        "X_test_raw":   X_test,
        "scaled":       scaled,
        "scaled_anim":  scaled_anim,
        "anim_data":    anim_data_full,
        "scaler":       scaler,
        "split":        split,
    }
