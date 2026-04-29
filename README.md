# WTI Crude Oil Futures — Deep Learning Price Forecasting

**Course:** ML in Finance  
**Dataset:** WTI Crude Oil Futures (CL=F) — daily OHLCV, 2018-01-01 to present (~2,090 rows)

---

## Project Structure

```
data/
    train/              train split CSV (written by build_dataset.py)
    dev/                dev split CSV
    test/               test split CSV
experiments/
    hyperparam_search/  grid search results JSON + comparison plot
    lookback_search/    look-back window experiment results JSON + plot
    loss_fn_search/     MAE vs MSE experiment results JSON
    training_logs/      JSON + TXT logs written by train.py
model/
    input_fn.py         data loading, scaling, sequence building
    model_fn.py         VanillaRNN, LSTMModel, GRUModel, TransformerForecaster
    training.py         train_step with ReduceLROnPlateau scheduler
    evaluation.py       MSE, MAE, directional accuracy, Sharpe ratio
    utils.py            helper functions
saved_models/           .pt weights + scaler.pkl (written by train.py)
build_dataset.py        fetch & split dataset from Yahoo Finance
train.py                full training run — all four DL models
evaluate.py             load saved models and print metrics
run_demo.py             interactive demo script with visualisation
search_hyperparams.py   grid search over model / lr / epochs / scheduler
search_lookback.py      experiment: compare look-back window sizes
search_loss_fn.py       experiment: compare MAE vs MSE training loss
synthesize_results.py   aggregate experiment JSON files into a CSV
main.ipynb              end-to-end interactive notebook
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

```bash
python build_dataset.py
```

Downloads `wti_crude_daily.csv` (WTI Crude Oil Futures, CL=F, daily OHLCV from 2018-01-01) and splits it into `data/train/`, `data/dev/`, `data/test/`.

### 3 — Run the demo (recommended entry point)

```bash
python run_demo.py
```

Loads saved weights from `saved_models/` if they exist, otherwise trains from scratch. Outputs a metrics table and saves `demo_results.png`.

| Flag | Effect |
|---|---|
| `--retrain` | Force fresh training even if weights exist |
| `--loss mse\|mae` | Training loss function (default: `mse`) |

### 4 — Full training run

```bash
python train.py               # MSE loss (default)
python train.py --loss mae    # MAE loss
```

Trains all four models for 50 epochs, saves weights to `saved_models/`, logs to `experiments/training_logs/`.

### 5 — Evaluate saved models

```bash
python evaluate.py
```

### 6 — Experiments

#### Hyperparameter search
```bash
python search_hyperparams.py                  # train and plot
python search_hyperparams.py --plot-only      # re-plot from existing results
```
Grid: model × lr × epochs × scheduler on/off.  
Results → `experiments/hyperparam_search/search_results.json`  
Plot → `experiments/hyperparam_search/hyperparam_comparison.png`

#### Look-back window comparison
```bash
python search_lookback.py                     # train and plot
python search_lookback.py --plot-only         # re-plot from existing results
```
Tests look-back windows of **5, 10, 20, 50** days across all models.  
Results → `experiments/lookback_search/lookback_results.json`  
Plot → `experiments/lookback_search/lookback_comparison.png`

#### MAE vs MSE loss function comparison
```bash
python search_loss_fn.py                      # train and print table
python search_loss_fn.py --results-only       # re-print from existing results
```
Trains each model with both MSE and MAE, evaluates with the matching metric plus directional accuracy and Sharpe ratio. Prints two side-by-side tables.  
Results → `experiments/loss_fn_search/loss_fn_results.json`

### 7 — Aggregate results

```bash
python synthesize_results.py
```

---

## Models

| Model | Architecture |
|---|---|
| **VanillaRNN** | 2-layer RNN → BatchNorm1d → Linear |
| **LSTM** | 2-layer LSTM → BatchNorm1d → Linear |
| **GRU** | 2-layer GRU → BatchNorm1d → Linear |
| **Transformer** | Linear input projection → sinusoidal PE → 2× decoder layer (8-head causal self-attention + FFN) → last-token readout → Linear |

**Shared config (all DL models):**

| Hyperparameter | Value |
|---|---|
| Hidden / d_model size | 64 |
| Layers | 2 |
| Dropout | 0.2 |
| Input features | Open, High, Low, Close, Volume (5) |
| Look-back window | 10 trading days |
| Target | Next-day Close (scaled) |
| Batch size | 64 |
| Epochs | 50 |
| Optimiser | Adam (lr = 1e-2) |
| LR scheduler | `ReduceLROnPlateau` (factor=0.5, patience=5, min_lr=1e-6) — enabled by default |
| Loss | MSE (default) or MAE via `--loss mae` |

**Transformer specifics:**  
Decoder-only (GPT-style) with a causal upper-triangular mask applied at every layer, so each position only attends to itself and earlier steps. A zero-filled prediction slot is appended to the input sequence; the model reads the forecast from that final position.

---

## ML Techniques

| Technique | Where applied |
|---|---|
| **StandardScaler** normalisation | `model/input_fn.py` — fitted on train data only to prevent leakage |
| **Sliding-window sequences** | 10-day window → predict next-day Close |
| **Causal (masked) self-attention** | Transformer decoder layer — no future leakage |
| **Sinusoidal positional encoding** | Injects chronological order into Transformer input |
| **Zero-filled prediction slot** | Appended to Transformer input; forecast read from last token |
| **BatchNorm1d** on final hidden state | Stabilises RNN / LSTM / GRU training |
| **Dropout (0.2)** | All recurrent layers and Transformer encoder layers |
| **Adam optimiser** | Adaptive learning-rate optimisation |
| **ReduceLROnPlateau** | Halves LR after 5 epochs without improvement; floor 1e-6 |
| **MSE loss** | Default training objective; penalises large errors more heavily |
| **MAE loss** | Robust to outliers; selectable via `--loss mae` |
| **DirectionalMSELoss** | Custom loss: multiplies wrong-direction squared errors by `(1 + α=10)` |
| **Walk-forward ARIMA** | Classical linear baseline; refitted at every test step |
| **Directional accuracy** | % of days where predicted direction matches actual (random baseline = 50%) |
| **Annualised Sharpe ratio** | Risk-adjusted return of a long/short strategy driven by model signals |

---

## Dataset

`wti_crude_daily.csv` — daily OHLCV for WTI Crude Oil Futures (CL=F) from 2018-01-01, fetched via `yfinance`.

**Chronological split (no shuffling):**

| Split | Proportion | Purpose |
|---|---|---|
| Train | 80% | Model training |
| Dev | 20% | Hyperparameter tuning |
| Animation / test | last 30 rows | Visual demo only |

---

## Overleaf Report

LaTeX source: `report/main.tex`. Figures: `report/figures/`.

```bash
python generate_figures.py   # regenerate all figures
zip -r report.zip report/    # zip for Overleaf upload
```
