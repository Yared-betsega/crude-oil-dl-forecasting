baseline_plain_mae/
    params.json
    metrics_eval.json
    metrics_dev.json

directional_loss_alpha10/
    params.json
    metrics_eval.json
    metrics_dev.json

Each sub-folder holds one experiment run.
`params.json`       — hyperparameters used for that run
`metrics_eval.json` — test-set metrics (MAE, Dir Acc, Sharpe)
`metrics_dev.json`  — dev-set metrics recorded during training
