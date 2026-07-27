"""
ThumaCheck — MLFlow Experiment Tracker
=======================================

Wrapper minimaliste pour logger les entraînements dans MLFlow.
Stocke métriques, paramètres et artefacts localement (./mlruns/).

Usage::

    from monitoring.mlflow_tracker import track_experiment

    with track_experiment("V5_TF-IDF_LogReg", params={"C": 1.0}) as run:
        # ... entraînement ...
        run.log_metrics({"f1": 0.913, "accuracy": 0.91})
        run.log_artifact("models/model_expert_v5.pkl")
"""

import os
from contextlib import contextmanager

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

EXPERIMENT_NAME = "ThumaCheck"
TRACKING_URI = os.path.join(
    os.path.dirname(__file__), '..', '..', 'mlruns'
)


class _RunProxy:
    """Proxy simplifié autour d'un run MLFlow actif."""

    def __init__(self, run):
        self._run = run

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    def log_params(self, params: dict):
        for k, v in params.items():
            mlflow.log_param(k, str(v))

    def log_artifact(self, path: str):
        if os.path.exists(path):
            mlflow.log_artifact(path)

    @property
    def run_id(self) -> str:
        return self._run.info.run_id


class _NoopProxy:
    """Fallback silencieux si MLFlow n'est pas installé."""

    def log_metrics(self, metrics: dict):
        pass

    def log_params(self, params: dict):
        pass

    def log_artifact(self, path: str):
        pass

    @property
    def run_id(self) -> str:
        return "noop"


@contextmanager
def track_experiment(run_name: str, params: dict | None = None):
    """Context manager pour tracker un entraînement.

    Parameters
    ----------
    run_name : str
        Nom du run (ex: "V5_TF-IDF_LogReg").
    params : dict, optional
        Hyperparamètres à logger.

    Yields
    ------
    _RunProxy ou _NoopProxy
    """
    if not MLFLOW_AVAILABLE:
        yield _NoopProxy()
        return

    mlflow.set_tracking_uri(f"file://{os.path.abspath(TRACKING_URI)}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        proxy = _RunProxy(run)
        if params:
            proxy.log_params(params)
        yield proxy
