import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np

from src.evaluation.evaluation_metrics import BusinessEvaluationFramework  # Update import path

@pytest.fixture
def mock_data():
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 240, 310])
    units_sold = np.array([10, 15, 20, 25, 30])
    return y_true, y_pred, units_sold

def test_calculate_revenue_impact(mock_data):
    y_true, y_pred, units_sold = mock_data
    evaluator = BusinessEvaluationFramework()
    result = evaluator.calculate_revenue_impact(y_true, y_pred, units_sold)
    assert "actual_revenue" in result
    assert "predicted_revenue" in result
    assert "revenue_impact_pct" in result

def test_calculate_pricing_accuracy_metrics(mock_data):
    y_true, y_pred, _ = mock_data
    evaluator = BusinessEvaluationFramework()
    result = evaluator.calculate_pricing_accuracy_metrics(y_true, y_pred)
    assert "MAE" in result
    assert "MAPE" in result
    assert "R2" in result
    assert "Directional_Accuracy" in result

def test_generate_business_report(mock_data):
    y_true, y_pred, units_sold = mock_data
    evaluator = BusinessEvaluationFramework()
    report = evaluator.generate_business_report(y_true, y_pred, units_sold)
    assert "Revenue_Impact" in report
    assert "Accuracy_Metrics" in report
    assert "Summary" in report