# tracking_utils

This package is MILES' common interface for experiment tracking. `tracking.py`
is the public entry point; its `init_tracking`, `log`,
`define_step_key_metric_group`, and `finish_tracking` functions fan work out to
the active backends.

Use these entry points rather than importing backend modules directly.

The supported backends are W&B, TensorBoard, MLflow, Prometheus, CI history,
and the Miles dashboard. Backend-specific helpers stay in their own modules.
Each backend is enabled through its corresponding CLI flag.

The package also includes structured-logging helpers in `structured_log.py`.

The two `wandb.init` call sites, inside `init_wandb_primary` and
`init_wandb_secondary` in `wandb_utils.py`, use a 300-second timeout.
