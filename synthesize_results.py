import os
import json
import glob

import pandas as pd

EXPERIMENTS_DIR = "experiments"


def load_all_metrics(base_dir: str = EXPERIMENTS_DIR) -> pd.DataFrame:
    rows = []
    for metrics_file in glob.glob(os.path.join(base_dir, "**", "metrics_eval.json"),
                                   recursive=True):
        params_file = os.path.join(os.path.dirname(metrics_file), "params.json")
        with open(metrics_file) as f:
            metrics = json.load(f)
        params = {}
        if os.path.exists(params_file):
            with open(params_file) as f:
                params = json.load(f)
        rows.append({**params, **metrics,
                     "experiment": os.path.dirname(metrics_file)})

    search_file = os.path.join(base_dir, "hyperparam_search", "search_results.json")
    if os.path.exists(search_file):
        with open(search_file) as f:
            rows.extend(json.load(f))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def synthesize():
    df = load_all_metrics()
    if df.empty:
        print("No experiment results found under experiments/.")
        print("Run train.py or search_hyperparams.py first.")
        return

    metric_cols = [c for c in ["mae", "dir_acc", "sharpe"] if c in df.columns]
    print("\n=== All Experiment Results ===")
    print(df.to_string(index=False))

    if "sharpe" in df.columns:
        best = df.loc[df["sharpe"].idxmax()]
        print(f"\nBest run (Sharpe={best['sharpe']:.4f}):")
        print(best.to_string())

    out_path = os.path.join(EXPERIMENTS_DIR, "all_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSummary saved → {out_path}")


if __name__ == "__main__":
    synthesize()
