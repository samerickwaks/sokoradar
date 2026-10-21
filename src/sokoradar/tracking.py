"""Optional MLflow experiment tracking (local ./mlruns). Silently does nothing without MLflow."""

from __future__ import annotations

import logging
import os

from .config import ROOT

log = logging.getLogger(__name__)


def log_run(name: str, params: dict, metrics: dict, artifacts: list | None = None) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("sokoradar")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            for a in artifacts or []:
                mlflow.log_artifact(str(a))
    except Exception as e:  # tracking must never break the pipeline
        log.warning("MLflow logging skipped: %s", e)
