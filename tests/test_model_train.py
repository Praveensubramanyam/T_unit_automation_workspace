import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.models.model_training import MLExperimentFramework  # Update import path

@pytest.fixture
def ml_framework():
    return MLExperimentFramework()

@pytest.fixture
def mock_data():
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(np.random.rand(100))
    return X, y

def test_setup_models(ml_framework):
    ml_framework.setup_models()
    assert "random_forest" in ml_framework.models
    assert "xgboost" in ml_framework.models
    assert "lightgbm" in ml_framework.models

def test_get_hyperparameter_space(ml_framework):
    space = ml_framework.get_hyperparameter_space("xgboost")
    assert "n_estimators" in space
    assert "max_depth" in space

def test_hyperparameter_tuning_optuna(ml_framework, mock_data):
    X, y = mock_data
    ml_framework.setup_models()
    best_params = ml_framework.hyperparameter_tuning_optuna("xgboost", X, y, X, y, n_trials=1)
    assert "n_estimators" in best_params
    assert "max_depth" in best_params

def test_cross_validate_models(ml_framework, mock_data):
    X, y = mock_data
    ml_framework.setup_models()
    ml_framework.cross_validate_models(X, y, cv_folds=3)
    assert "random_forest" in ml_framework.results
    assert "xgboost" in ml_framework.results

def test_train_best_models(ml_framework, mock_data):
    X, y = mock_data
    ml_framework.setup_models()
    ml_framework.train_best_models(X, y, X, y)
    assert "random_forest" in ml_framework.models
    assert "xgboost" in ml_framework.models
