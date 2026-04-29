"""
Experiment: compare MAE vs MSE training loss across all models.

Each model is trained twice — once with MSE and once with MAE — and evaluated
on the same test set with MSE, MAE, directional accuracy, and Sharpe ratio.

Results are saved to:
    experiments/loss_fn_search/loss_fn_results.json

Usage:
    python search_loss_fn.py [--epochs N]
    python search_loss_fn.py --plot-only
"""

import os
import json
import argparse

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.input_fn   import get_data_loaders, TARGET_COL
from model.model_fn   import build_model, MODEL_REGISTRY
from model.training   import train_step, LOSS_CHOICES
from model.evaluation import full_eval
from model.utils      import directional_accuracy

CSV_PATH    = "wti_crude_daily.csv"
RESULTS_DIR = os.path.join("experiments", "loss_fn_search")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOSS_FNS = ["mse", "mae"]


def run(epochs: int = 50):
    data         = get_data_loaders(CSV_PATH)
    scaler       = data["scaler"]
    train_loader = data["train_loader"]
    X_train_t    = data["X_train"]
    y_train_t    = data["y_train"]
    X_train_raw  = data["X_train_raw"]
    X_test_t     = data["X_test"]
    y_test_t     = data["y_test"]
    X_test_raw   = data["X_test_raw"]

    all_results = []

    for loss_fn in LOSS_FNS:
        print(f"\n{'='*55}")
        print(f"  Training loss: {loss_fn.upper()}")
        print("=" * 55)

        for model_name in MODEL_REGISTRY:
            print(f"  [{model_name}]")
            model = build_model(model_name)
            train_step(model, train_loader,
                       epochs=epochs, loss=loss_fn, verbose=False)

            # Evaluate on train set for both MAE and MSE
            model.eval()
            with torch.no_grad():
                train_preds  = model(X_train_t).numpy().flatten()
            train_actual = y_train_t.numpy().flatten()
            train_mse = mean_squared_error(train_actual, train_preds)
            train_mae = mean_absolute_error(train_actual, train_preds)

            # Evaluate on test set
            metrics = full_eval(
                model, X_test_t, y_test_t, X_test_raw, scaler, TARGET_COL
            )

            row = {
                "loss_fn":   loss_fn,
                "model":     model_name,
                "train_mae": round(train_mae, 6),
                "train_mse": round(train_mse, 6),
                "test_mae":  round(metrics.get("mae", float("nan")), 6),
                "test_mse":  round(metrics["mse"],                   6),
                "dir_acc":   round(metrics["dir_acc"],               4),
                "sharpe":    round(metrics["sharpe"],                4),
            }
            all_results.append(row)

    out_path = os.path.join(RESULTS_DIR, "loss_fn_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    _print_table(all_results)
    return all_results


def _print_table(results):
    cw = {"model": 13, "num": 12, "dir": 11, "sharpe": 8}
    header_mse = (
        f"{'Model':<{cw['model']}} "
        f"{'Train MSE':>{cw['num']}} {'Test MSE':>{cw['num']}} "
        f"{'Dir Accu %':>{cw['dir']}} {'Sharpe':>{cw['sharpe']}}"
    )
    header_mae = (
        f"{'Model':<{cw['model']}} "
        f"{'Train MAE':>{cw['num']}} {'Test MAE':>{cw['num']}} "
        f"{'Dir Accu %':>{cw['dir']}} {'Sharpe':>{cw['sharpe']}}"
    )
    sep_mse = "-" * len(header_mse)
    sep_mae = "-" * len(header_mae)

    print(f"\nTrained with MSE")
    print(sep_mse)
    print(header_mse)
    print(sep_mse)
    for r in [r for r in results if r["loss_fn"] == "mse"]:
        print(
            f"{r['model']:<{cw['model']}} "
            f"{r['train_mse']:>{cw['num']}.6f} {r['test_mse']:>{cw['num']}.6f} "
            f"{r['dir_acc']*100:>{cw['dir']}.1f} {r['sharpe']:>{cw['sharpe']}.4f}"
        )
    print(sep_mse)

    print(f"\nTrained with MAE")
    print(sep_mae)
    print(header_mae)
    print(sep_mae)
    for r in [r for r in results if r["loss_fn"] == "mae"]:
        print(
            f"{r['model']:<{cw['model']}} "
            f"{r['train_mae']:>{cw['num']}.6f} {r['test_mae']:>{cw['num']}.6f} "
            f"{r['dir_acc']*100:>{cw['dir']}.1f} {r['sharpe']:>{cw['sharpe']}.4f}"
        )
    print(sep_mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE vs MSE loss function comparison")
    parser.add_argument("--epochs",    type=int, default=50,
                        help="Training epochs per model (default: 50)")
    parser.add_argument("--results-only", action="store_true",
                        help="Skip training; re-print table from existing results JSON")
    args = parser.parse_args()

    if args.results_only:
        with open(os.path.join(RESULTS_DIR, "loss_fn_results.json")) as f:
            results = json.load(f)
        _print_table(results)
    else:
        run(epochs=args.epochs)
