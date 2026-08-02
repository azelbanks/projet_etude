"""
Tests du tracker MLFlow.

Ce module etait a 0 % de couverture — la seule brique MLOps du depot n'etait
pas testee. Les tests couvrent les deux chemins : MLFlow disponible (via mock,
aucun serveur requis) et MLFlow absent (fallback silencieux).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from monitoring import mlflow_tracker
from monitoring.mlflow_tracker import _NoopProxy, _RunProxy, track_experiment


class TestNoopProxy:
    """MLFlow absent : le tracker ne doit jamais faire echouer un entrainement."""

    def test_log_metrics_is_silent(self):
        assert _NoopProxy().log_metrics({"f1": 0.9}) is None

    def test_log_params_is_silent(self):
        assert _NoopProxy().log_params({"C": 1.0}) is None

    def test_log_artifact_is_silent(self):
        assert _NoopProxy().log_artifact("/chemin/inexistant.pkl") is None

    def test_run_id_is_noop(self):
        assert _NoopProxy().run_id == "noop"

    def test_context_manager_yields_noop_when_unavailable(self):
        with patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", False):
            with track_experiment("run_test") as run:
                assert isinstance(run, _NoopProxy)
                run.log_metrics({"f1": 0.9})  # ne doit pas lever

    def test_params_ignored_when_unavailable(self):
        """Les params passes au context manager ne doivent pas lever non plus."""
        with patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", False):
            with track_experiment("run_test", params={"C": 1.0}) as run:
                assert run.run_id == "noop"


class TestRunProxy:
    """MLFlow disponible : chaque appel doit etre relaye a l'API mlflow."""

    def test_log_metrics_forwards_each_key(self):
        fake = MagicMock()
        with patch.object(mlflow_tracker, "mlflow", fake):
            _RunProxy(MagicMock()).log_metrics({"f1": 0.913, "accuracy": 0.91})
        assert fake.log_metric.call_count == 2
        fake.log_metric.assert_any_call("f1", 0.913)
        fake.log_metric.assert_any_call("accuracy", 0.91)

    def test_log_params_stringifies_values(self):
        """log_param recoit des chaines : MLFlow refuse certains types natifs."""
        fake = MagicMock()
        with patch.object(mlflow_tracker, "mlflow", fake):
            _RunProxy(MagicMock()).log_params({"C": 1.0, "solver": "lbfgs"})
        fake.log_param.assert_any_call("C", "1.0")
        fake.log_param.assert_any_call("solver", "lbfgs")

    def test_log_artifact_skips_missing_file(self):
        fake = MagicMock()
        with patch.object(mlflow_tracker, "mlflow", fake):
            _RunProxy(MagicMock()).log_artifact("/chemin/vraiment/inexistant.pkl")
        fake.log_artifact.assert_not_called()

    def test_log_artifact_forwards_existing_file(self, tmp_path):
        artefact = tmp_path / "model.pkl"
        artefact.write_bytes(b"contenu")
        fake = MagicMock()
        with patch.object(mlflow_tracker, "mlflow", fake):
            _RunProxy(MagicMock()).log_artifact(str(artefact))
        fake.log_artifact.assert_called_once_with(str(artefact))

    def test_run_id_reads_mlflow_run_info(self):
        run = MagicMock()
        run.info.run_id = "abc123"
        assert _RunProxy(run).run_id == "abc123"

    def test_empty_metrics_forwards_nothing(self):
        fake = MagicMock()
        with patch.object(mlflow_tracker, "mlflow", fake):
            _RunProxy(MagicMock()).log_metrics({})
        fake.log_metric.assert_not_called()


class TestTrackExperiment:
    """Cycle complet du context manager avec MLFlow mocke."""

    @staticmethod
    def _fake_mlflow():
        fake = MagicMock()
        run = MagicMock()
        run.info.run_id = "run-42"
        fake.start_run.return_value.__enter__.return_value = run
        return fake

    def test_sets_tracking_uri_and_experiment(self):
        fake = self._fake_mlflow()
        with (
            patch.object(mlflow_tracker, "mlflow", fake),
            patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", True),
        ):
            with track_experiment("V5_TF-IDF_LogReg"):
                pass
        fake.set_tracking_uri.assert_called_once()
        assert fake.set_tracking_uri.call_args[0][0].startswith("file://")
        fake.set_experiment.assert_called_once_with(mlflow_tracker.EXPERIMENT_NAME)

    def test_run_name_is_passed_through(self):
        fake = self._fake_mlflow()
        with (
            patch.object(mlflow_tracker, "mlflow", fake),
            patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", True),
        ):
            with track_experiment("mon_run"):
                pass
        fake.start_run.assert_called_once_with(run_name="mon_run")

    def test_params_logged_before_body_runs(self):
        fake = self._fake_mlflow()
        with (
            patch.object(mlflow_tracker, "mlflow", fake),
            patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", True),
        ):
            with track_experiment("run", params={"C": 1.0, "max_iter": 500}):
                pass
        fake.log_param.assert_any_call("C", "1.0")
        fake.log_param.assert_any_call("max_iter", "500")

    def test_yields_run_proxy_with_id(self):
        fake = self._fake_mlflow()
        with (
            patch.object(mlflow_tracker, "mlflow", fake),
            patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", True),
        ):
            with track_experiment("run") as proxy:
                assert isinstance(proxy, _RunProxy)
                assert proxy.run_id == "run-42"

    def test_metrics_logged_from_body(self):
        fake = self._fake_mlflow()
        with (
            patch.object(mlflow_tracker, "mlflow", fake),
            patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", True),
        ):
            with track_experiment("run") as proxy:
                proxy.log_metrics({"f1": 0.913})
        fake.log_metric.assert_called_once_with("f1", 0.913)

    def test_exception_in_body_propagates(self):
        """Une erreur d'entrainement ne doit pas etre avalee par le tracker."""
        fake = self._fake_mlflow()
        with (
            patch.object(mlflow_tracker, "mlflow", fake),
            patch.object(mlflow_tracker, "MLFLOW_AVAILABLE", True),
        ):
            with pytest.raises(ValueError, match="echec entrainement"):
                with track_experiment("run"):
                    raise ValueError("echec entrainement")


class TestModuleContract:
    def test_experiment_name_is_defined(self):
        assert mlflow_tracker.EXPERIMENT_NAME == "ThumaCheck"

    def test_tracking_uri_points_to_mlruns(self):
        assert mlflow_tracker.TRACKING_URI.endswith("mlruns")

    def test_availability_flag_is_boolean(self):
        assert isinstance(mlflow_tracker.MLFLOW_AVAILABLE, bool)
