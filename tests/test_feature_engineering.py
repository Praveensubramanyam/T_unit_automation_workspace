import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import AdvancedPricingFeatureEngineer

@pytest.fixture
def mock_data():
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "MRP_x": pd.Series(np.random.uniform(100, 200, size=10)),
        "Brand": pd.Series(["BrandA"] * 5 + ["BrandB"] * 5),
        "FC_ID": pd.Series(["FC1"] * 5 + ["FC2"] * 5),
        "SellingPrice": pd.Series(np.random.uniform(90, 190, size=10)),
        "Demand": pd.Series(np.random.randint(1, 10, size=10)),
        "StockStart": pd.Series(np.random.randint(10, 50, size=10)),
        "IsMetro": pd.Series(np.random.choice([0, 1], size=10)),
        "LeadTimeFloat": pd.Series(np.random.uniform(1, 5, size=10)),
    }
    return pd.DataFrame(data)

@pytest.fixture
def feature_engineer():
    fe = AdvancedPricingFeatureEngineer()
    fe.target_column = "SellingPrice"  # ✅ Set target column manually
    return fe

def test_advanced_feature_engineering(feature_engineer, mock_data):
    df = mock_data
    result = feature_engineer.advanced_feature_engineering(df)
    assert "Log_MRP" in result.columns
    assert "MRP_Category" in result.columns
    assert "IsMetroMarket" in result.columns
    assert "Is_Premium_Brand" in result.columns
    assert "Is_Month_Start" in result.columns
    assert "Is_Month_End" in result.columns
    assert "Is_New_Product" in result.columns
    assert "Is_Mature_Product" in result.columns

def test_encode_categorical_features(feature_engineer, mock_data):
    df = mock_data
    result = feature_engineer.encode_categorical_features(df)
    assert "Brand_encoded" in result.columns
    assert "FC_ID_encoded" in result.columns

def test_prepare_features_target(feature_engineer, mock_data):
    df = mock_data
    df_engineered = feature_engineer.advanced_feature_engineering(df)
    X, y = feature_engineer.prepare_features_target(df_engineered)
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
    assert "Log_MRP" in X.columns

def test_fit_transform(feature_engineer, mock_data):
    df = mock_data
    X, y, df_encoded = feature_engineer.fit_transform(df)
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
    assert "Log_MRP" in X.columns

def test_transform(feature_engineer, mock_data):
    df = mock_data
    feature_engineer.fit_transform(df)  # Fit first
    X_new, df_encoded = feature_engineer.transform(df)
    assert X_new.shape[0] == df.shape[0]
    assert "Log_MRP" in X_new.columns
