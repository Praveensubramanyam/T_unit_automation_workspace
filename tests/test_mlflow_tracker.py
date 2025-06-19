import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.models.mlflow_tracker import MLflowPricingTracker  # Update import path

@pytest.fixture
def tracker():
    return MLflowPricingTracker()

def test_setup_experiment(tracker):
    with patch("mlflow.set_experiment") as mock_set_experiment:
        tracker.setup_experiment()
        mock_set_experiment.assert_called_once()

def test_log_feature_engineering_run(tracker):
    feature_stats = {"total_features": 10, "product_features": 5}
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    with patch("mlflow.start_run"), patch("mlflow.log_param"), patch("mlflow.log_metric"):
        tracker.log_feature_engineering_run(feature_stats, feature_names)

def test_log_model_experiment(tracker):
    model = MagicMock()
    model_name = "xgboost"
    params = {"n_estimators": 100}
    metrics = {"MAPE": 0.1, "R2": 0.9}
    with patch("mlflow.start_run"), patch("mlflow.log_param"), patch("mlflow.log_metric"), patch("mlflow.xgboost.log_model"):
        tracker.log_model_experiment(model, model_name, params, metrics)

def test_compare_model_runs(tracker):
    mock_runs_df = pd.DataFrame([
        {"run_id": "123", "metrics.MAPE": 0.1},
        {"run_id": "456", "metrics.MAPE": 0.2}
    ])

    with patch("mlflow.get_experiment_by_name") as mock_get_exp, \
         patch("mlflow.search_runs") as mock_search:
        mock_get_exp.return_value = MagicMock(experiment_id="1")
        mock_search.return_value = mock_runs_df

        result = tracker.compare_model_runs()
        assert result is not None
        assert "best_run_id" in result
        assert result["best_run_id"] == "123"

def test_load_best_model(tracker):
    with patch("mlflow.pyfunc.load_model") as mock_load:
        mock_load.return_value = MagicMock()
        model = tracker.load_best_model("123")
        assert model is not None
