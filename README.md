# WTI Crude Oil Futures — Deep Learning Price Forecasting

**Course:** ML in Finance  
**Dataset:** WTI Crude Oil Futures (CL=F) — daily OHLCV, 2018-01-01 to present (~2,090 rows)

---

## Project Structure

```
data/
    train/          train split CSV (written by build_dataset.py)
    dev/            dev split CSV
    test/
        sample_test_data.csv    last 100 rows of the full dataset
experiments/
    training_logs/  JSON + TXT logs written by train.py
model/
    input_fn.py     data loading, scaling, sequence building
    model_fn.py     VanillaRNN, LSTMModel, GRUModel, TransformerForecaster
    training.py     train_step, DirectionalMSELoss (logs every 5 epochs)
    evaluation.py   MSE, directional accuracy, Sharpe ratio
    utils.py        helper functions
saved_models/       .pt weights + scaler.pkl (written by train.py)
build_dataset.py    fetch & split dataset from Yahoo Finance
train.py            full training run — all four DL models
evaluate.py         load saved models and print metrics
run_demo.py         single-file TA demo script
search_hyperparams.py   grid search over model / lr / epochs
synthesize_results.py   aggregate experiment JSONs into a CSV
main.ipynb          end-to-end interactive notebook
```

---

## Quick Start

### 1 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch scikit-learn pandas numpy matplotlib statsmodels yfinance
```

### 2 — Download the dataset

The full dataset will be downloaded as `wti_crude_daily.csv`.  

```bash
python build_dataset.py
```

This writes `wti_crude_daily.csv` and splits it into `data/train/`, `data/dev/`, `data/test/`.

### 3 — Run the TA demo (recommended entry point)

```bash
python run_demo.py
```

- Loads saved model weights from `saved_models/` if they exist.  
- Falls back to training from scratch if weights are missing.  
- Outputs a metrics table to the terminal and saves `demo_results.png`.

Options:

| Flag | Effect |
|---|---|
| `--retrain` | Force fresh training even if weights exist |
| `--sample`  | Use the 100-row sample CSV instead of the full dataset |
| `--loss`    | Training loss function: `mse` (default) or `mae` |

```bash
python run_demo.py --retrain                  # re-train everything
python run_demo.py --sample                   # quick smoke-test on sample data
python run_demo.py --retrain --loss mae       # re-train with MAE loss
```

### 4 — Full training run (saves logs)

```bash
python train.py
```

- Trains RNN, LSTM, GRU, Transformer for 50 epochs each.  
- Logs the training loss every **5 epochs** to the terminal.  
- Saves `experiments/training_logs/training_log_<timestamp>.json` and a human-readable `summary_<timestamp>.txt`.  
- Saves model weights to `saved_models/`.

Use `--loss` to switch the training loss function:

```bash
python train.py               # MSE (default)
python train.py --loss mae    # MAE
```

### 5 — Evaluate saved models

```bash
python evaluate.py
```

### 6 — Hyperparameter search

```bash
python search_hyperparams.py                 # MSE (default)
python search_hyperparams.py --loss mae      # run search with MAE loss
```

Results are written to `experiments/hyperparam_search/search_results.json`.

### 7 — Aggregate experiment results

```bash
python synthesize_results.py
```

---

## Models

| Model | Architecture | Parameters |
|---|---|---|
| **VanillaRNN** | 2-layer RNN + BatchNorm + Linear | ~18 k |
| **LSTM** | 2-layer LSTM + BatchNorm + Linear | ~69 k |
| **GRU** | 2-layer GRU + BatchNorm + Linear | ~52 k |
| **Transformer** | Input projection + sinusoidal PE + 2× TransformerEncoderLayer (4 heads) + mean-pool + Linear | ~66 k |
| **ARIMA(5,1,0)** | Walk-forward autoregressive baseline | — |

**Training config (all DL models):**  
- Optimiser: Adam (`lr=1e-3`)  
- Loss: MSE (`nn.MSELoss`) — configurable via `--loss {mse,mae}`  
- Epochs: 50 | Batch size: 64 | Dropout: 0.2  
- Input: 60-day sliding window of [Open, High, Low, Close, Volume] scaled with `StandardScaler`  
- Target: next-day Close (scaled)

---

## Results

| Model | MSE (scaled) | Directional Acc. | Sharpe (ann.) |
|---|---|---|---|
| RNN | 0.1048 | 52.1% | 0.792 |
| LSTM | 0.0760 | 53.4% | **0.991** |
| GRU | 0.0710 | 51.4% | −0.130 |
| Transformer | 0.0754 | **55.4%** | 0.927 |
| ARIMA(5,1,0) | **0.0690** | 52.6% | 0.269 |

> These values are from the previous MAE-trained run. Re-run `python train.py` to update with MSE-trained results.

- **MSE:** penalises large prediction errors more heavily than MAE.  
- **Directional accuracy:** all models sit just above the 50% random baseline; the Transformer edges ahead at 55.4%.  
- **Sharpe ratio:** LSTM (0.991) and Transformer (0.927) both approach the 1.0 threshold considered attractive in systematic strategies. GRU's negative Sharpe (−0.130) shows that low prediction error does not imply profitable directional calls.

---

## ML Techniques

| Technique | Where applied |
|---|---|
| **StandardScaler** normalisation | `model/input_fn.py` — fitted on train/dev only to prevent data leakage |
| **Sliding-window sequences** | 60 trading days → predict next-day Close |
| **Vanishing-gradient mitigation** | LSTM (3 gates) and GRU (2 gates) vs. vanilla RNN |
| **Multi-head self-attention** | Transformer encoder captures global temporal dependencies simultaneously |
| **Sinusoidal positional encoding** | Injects chronological order into the attention-based model |
| **BatchNorm1d** on final hidden state | Stabilises RNN/LSTM/GRU training |
| **Dropout (0.2)** | Applied in all recurrent layers and the Transformer encoder |
| **Adam optimiser** | Adaptive learning-rate optimisation |
| **MSE loss** | Penalises large errors more heavily than MAE; default training objective (`--loss mse`) |
| **MAE loss** | Robust to outliers; selectable via `--loss mae` |
| **Directional penalty loss** | Custom `DirectionalMSELoss`: multiplies wrong-direction squared errors by `(1 + α)` |
| **Walk-forward ARIMA** | Classical linear baseline; refitted at every test step to avoid look-ahead bias |
| **Directional accuracy** | % of days where predicted direction matches actual (random = 50%) |
| **Annualised Sharpe ratio** | Risk-adjusted return of a long/short strategy driven by model direction calls |

---

## Overleaf Report

The LaTeX source lives in `report/main.tex`.
All figures are pre-generated PDFs in `report/figures/`.

### Uploading to Overleaf

1. **Generate figures** (skip if already done):
   ```bash
   python generate_figures.py
   ```
2. **Zip the report folder:**
   ```bash
   zip -r report.zip report/
   ```
3. On [overleaf.com](https://www.overleaf.com):
   - Click **New Project → Upload Project**.
   - Upload `report.zip`.
   - Set the main document to `main.tex`.
   - Click **Recompile** — the paper will build immediately.

> To refresh figures after retraining, re-run `python generate_figures.py`
> and re-upload `report/figures/` via Overleaf's file manager.

---



1. All models learn a **persistence-like strategy** under plain MSE loss — the near-random-walk nature of crude oil futures means "predict ≈ yesterday's price" already minimises average error.  
2. The **Transformer's global attention** gives it a small but consistent directional edge (55.4%) over recurrent architectures.  
3. The **directional penalty loss** (`α=10`) successfully shifts models away from pure persistence toward directional correctness.  
4. The gap between MSE rankings and Sharpe rankings confirms that financial forecasting requires **multiple evaluation metrics**.

---

## Dataset

`wti_crude_daily.csv` — 2,091 rows, daily OHLCV for WTI Crude Oil Futures (CL=F) from 2018-01-01 fetched via `yfinance`.  
A 100-row sample is available at `data/test/sample_test_data.csv`.

**Chronological split (no shuffling):**

| Split | Rows | Purpose |
|---|---|---|
| Train (80%) | ~1,368 | Model training |
| Dev  (20%) | ~342 | Hyperparameter tuning |
| Animation | last 30 | Visual demo only |
