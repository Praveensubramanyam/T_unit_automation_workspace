import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
import pandas as pd
import numpy as np
from src.features.time_series import PricingTimeSeriesAnalyzer  # Update import path

@pytest.fixture
def mock_data():
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
        "SellingPrice": np.random.uniform(100, 200, size=100),
        "UnitsSold": np.random.randint(1, 10, size=100),
        "Brand": ["BrandA"] * 50 + ["BrandB"] * 50,
        "FC_ID": ["FC1"] * 50 + ["FC2"] * 50,
    }
    return pd.DataFrame(data)

@pytest.fixture
def ts_analyzer():
    return PricingTimeSeriesAnalyzer()

def test_decompose_time_series(ts_analyzer, mock_data):
    df = mock_data
    result = ts_analyzer.decompose_time_series(df, "SellingPrice")
    assert hasattr(result, "trend")
    assert hasattr(result, "seasonal")
    assert hasattr(result, "resid")

def test_detect_seasonality(ts_analyzer, mock_data):
    df = mock_data
    result = ts_analyzer.detect_seasonality(df, "SellingPrice")
    assert "dominant_periods" in result
    assert "seasonal_strength" in result

def test_create_lag_features(ts_analyzer, mock_data):
    df = mock_data
    result = ts_analyzer.create_lag_features(df, "SellingPrice")
    assert "SellingPrice_lag_1" in result.columns
    assert "SellingPrice_lag_7" in result.columns
    assert "SellingPrice_lag_14" in result.columns

def test_forecast_demand(ts_analyzer, mock_data):
    df = mock_data
    result = ts_analyzer.forecast_demand(df)
    assert isinstance(result, dict)
    assert all("forecast" in v for v in result.values())
