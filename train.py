import os
import json
import argparse
import torch
import pickle
from datetime import datetime

from model.input_fn   import get_data_loaders, TARGET_COL
from model.model_fn   import build_model, MODEL_REGISTRY
from model.training   import train_step, LOSS_CHOICES
from model.evaluation import full_eval, print_metrics

MODELS_DIR = "saved_models"
LOGS_DIR   = "experiments/training_logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

CSV_PATH = "wti_crude_daily.csv"


def main():
    parser = argparse.ArgumentParser(description="Full training run")
    parser.add_argument("--loss", default="mse", choices=LOSS_CHOICES,
                        help="Training loss function (default: mse)")
    args = parser.parse_args()

    data = get_data_loaders(CSV_PATH)

    scaler       = data["scaler"]
    train_loader = data["train_loader"]
    X_test_t     = data["X_test"]
    y_test_t     = data["y_test"]
    X_test_raw   = data["X_test_raw"]

    print(f"Loss function: {args.loss.upper()}")

    all_logs = {}
    results  = {}

    for name in MODEL_REGISTRY:
        print(f"\n{'='*45}")
        print(f"  Training {name}")
        print("=" * 45)
        model   = build_model(name)
        history = train_step(model, train_loader, loss=args.loss)
        metrics = full_eval(model, X_test_t, y_test_t, X_test_raw, scaler, TARGET_COL)
        metrics["model"] = model
        results[name]    = metrics
        all_logs[name]   = history

        print(f"  MSE={metrics['mse']:.4f}  Dir={metrics['dir_acc']*100:.1f}%  "
              f"Sharpe={metrics['sharpe']:.3f}")

        save_path = os.path.join(MODELS_DIR, f"{name.lower()}_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  Saved → {save_path}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved → {scaler_path}")

    # Save training logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOGS_DIR, f"training_log_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"Training log saved → {log_path}")

    # Also write a human-readable summary
    summary_path = os.path.join(LOGS_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Training run: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        for name, history in all_logs.items():
            log_key = f"train_{args.loss.lower()}"
            f.write(f"Model: {name}\n")
            f.write(f"{'Epoch':>6}  {f'Train {args.loss.upper()}':>12}\n")
            f.write("-" * 22 + "\n")
            for entry in history:
                f.write(f"{entry['epoch']:>6}  {entry[log_key]:>12.6f}\n")
            r = results[name]
            f.write(f"\n  Test MSE    : {r['mse']:.4f}\n")
            f.write(f"  Dir Acc (%) : {r['dir_acc']*100:.1f}%\n")
            f.write(f"  Sharpe      : {r['sharpe']:.3f}\n\n")
    print(f"Summary saved      → {summary_path}")

    print("\n\n=== Final Results ===")
    print_metrics({k: v for k, v in results.items() if k != "model"})


if __name__ == "__main__":
    main()
