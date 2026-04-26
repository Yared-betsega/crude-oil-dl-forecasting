import numpy as np
from sklearn.preprocessing import StandardScaler

TARGET_COL = 3


def inv_close(scaled_vals: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    dummy = np.zeros((len(scaled_vals), 5), dtype=np.float32)
    dummy[:, TARGET_COL] = scaled_vals
    return scaler.inverse_transform(dummy)[:, TARGET_COL]


def directional_accuracy(preds: np.ndarray, actuals: np.ndarray,
                          prev_actuals: np.ndarray) -> float:
    actual_dir = np.sign(actuals - prev_actuals)
    pred_dir   = np.sign(preds   - prev_actuals)
    mask       = actual_dir != 0
    return float(np.mean(actual_dir[mask] == pred_dir[mask]))


def sharpe_ratio(preds_scaled: np.ndarray, prev_scaled: np.ndarray,
                 actuals_scaled: np.ndarray, scaler: StandardScaler,
                 annualize: int = 252) -> float:
    prev_usd   = inv_close(prev_scaled,    scaler)
    actual_usd = inv_close(actuals_scaled, scaler)
    pred_usd   = inv_close(preds_scaled,   scaler)

    daily_ret = (actual_usd - prev_usd) / np.abs(prev_usd)
    position  = np.sign(pred_usd - prev_usd)
    position[position == 0] = 1

    strategy_ret = position * daily_ret
    mean_ret = np.mean(strategy_ret)
    std_ret  = np.std(strategy_ret, ddof=1)
    if std_ret == 0:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(annualize))
