import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import pandas as pd
import numpy as np
from src.pipeline.dynamic_pricing_pipeline import DynamicPricingPipeline

@pytest.fixture
def pipeline():
    return DynamicPricingPipeline()

@pytest.fixture
def mock_data():
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
        "MRP_x": pd.Series(np.random.uniform(100, 200, size=100)),
        "Brand": pd.Series(["BrandA"] * 50 + ["BrandB"] * 50),
        "FC_ID": pd.Series(["FC1"] * 50 + ["FC2"] * 50),
        "SellingPrice": pd.Series(np.random.uniform(90, 190, size=100)),
        "Demand": pd.Series(np.random.randint(1, 10, size=100)),
        "StockStart": pd.Series(np.random.randint(10, 50, size=100)),
        "IsMetro": pd.Series(np.random.choice([0, 1], size=100)),  # ✅ Added
        "LeadTimeFloat": pd.Series(np.random.uniform(1, 5, size=100)),  # ✅ Added
    }
    return pd.DataFrame(data)

def test_load_and_preprocess_data(pipeline, mock_data):
    df = pipeline.load_and_preprocess_data(mock_data)
    assert df.shape[0] == mock_data.shape[0]

def test_run_feature_engineering(pipeline, mock_data):
    X, y, df_processed = pipeline.run_feature_engineering(mock_data)
    assert X.shape[0] == mock_data.shape[0]
    assert y.shape[0] == mock_data.shape[0]

def test_run_time_series_analysis(pipeline, mock_data):
    df, forecasts = pipeline.run_time_series_analysis(mock_data)
    assert df.shape[0] == mock_data.shape[0]
    assert isinstance(forecasts, dict)

def test_prepare_model_data(pipeline, mock_data):
    X, y, df_processed = pipeline.run_feature_engineering(mock_data)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_model_data(X, y)
    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert X_test.shape[0] > 0

def test_run_ml_experiments(pipeline, mock_data):
    X, y, df_processed = pipeline.run_feature_engineering(mock_data)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_model_data(X, y)
    best_model_name, directional_accuracy = pipeline.run_ml_experiments(X_train, X_val, X_test, y_train, y_val, y_test)
    assert best_model_name is not None

def test_generate_business_insights(pipeline, mock_data):
    X, y, df_processed = pipeline.run_feature_engineering(mock_data)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_model_data(X, y)
    pipeline.run_ml_experiments(X_train, X_val, X_test, y_train, y_val, y_test)
    report = pipeline.generate_business_insights(X_test, y_test, "xgboost", 90)
    assert report is not None
    assert "Revenue_Impact" in report

def test_predict_prices(pipeline, mock_data):
    X, y, df_processed = pipeline.run_feature_engineering(mock_data)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_model_data(X, y)
    pipeline.run_ml_experiments(X_train, X_val, X_test, y_train, y_val, y_test)
    predictions, result_df = pipeline.predict_prices(mock_data)
    assert len(predictions) == mock_data.shape[0]
    assert "Predicted_SellingPrice" in result_df.columns

def test_run_complete_pipeline(pipeline, mock_data):
    result = pipeline.run_complete_pipeline(mock_data)
    assert result is not None
    assert "best_model" in result
    assert "business_report" in result
