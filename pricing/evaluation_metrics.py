import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


class BusinessEvaluationFramework:
    """
    Business-specific evaluation metrics for pricing optimization
    """

    def __init__(self):
        self.business_metrics = {}

    def calculate_revenue_impact(self, y_true, y_pred, units_sold):
        """Calculate revenue impact of pricing predictions"""
        actual_revenue = np.sum(y_true * units_sold)
        predicted_revenue = np.sum(y_pred * units_sold)

        revenue_impact = (
            (predicted_revenue - actual_revenue) / actual_revenue * 100
        )

        return {
            "actual_revenue": actual_revenue,
            "predicted_revenue": predicted_revenue,
            "revenue_impact_pct": revenue_impact,
        }

    def calculate_pricing_accuracy_metrics(self, y_true, y_pred):
        """Calculate comprehensive pricing accuracy metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Directional accuracy
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = (
            np.mean(actual_direction == pred_direction) * 100
        )

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "Directional_Accuracy": directional_accuracy,
        }

    def calculate_inventory_efficiency_impact(
        self, predictions, actual_demand, stock_levels
    ):
        """Calculate inventory efficiency improvements from better pricing"""
        # Stock optimization score
        optimal_stock_ratio = np.mean(
            actual_demand / np.maximum(stock_levels, 1)
        )

        # Stockout prevention score
        stockout_risk_reduction = np.mean(
            np.maximum(0, stock_levels - actual_demand)
            / np.maximum(stock_levels, 1)
        )

        return {
            "stock_efficiency_ratio": optimal_stock_ratio,
            "stockout_risk_reduction": stockout_risk_reduction,
        }

    def calculate_profit_optimization_score(
        self, prices, costs, units_sold, margins
    ):
        """Calculate profit optimization effectiveness"""
        if len(costs) == 0:
            costs = prices * 0.7  # Assume 30% margin if costs not available

        actual_profit = np.sum((prices - costs) * units_sold)
        margin_pct = np.mean(margins) if len(margins) > 0 else 30

        return {
            "total_profit": actual_profit,
            "average_margin_pct": margin_pct,
            "profit_per_unit": (
                actual_profit / np.sum(units_sold)
                if np.sum(units_sold) > 0
                else 0
            ),
        }

    def generate_business_report(
        self, y_true, y_pred, units_sold, additional_data=None
    ):
        """Generate comprehensive business impact report"""
        # Calculate all metrics
        revenue_metrics = self.calculate_revenue_impact(
            y_true, y_pred, units_sold
        )
        accuracy_metrics = self.calculate_pricing_accuracy_metrics(
            y_true, y_pred
        )

        # Combine all metrics
        business_report = {
            "Revenue_Impact": revenue_metrics,
            "Accuracy_Metrics": accuracy_metrics,
            "Summary": {
                "Total_Transactions": len(y_true),
                "Total_Units_Sold": np.sum(units_sold),
                "Average_Price_Actual": np.mean(y_true),
                "Average_Price_Predicted": np.mean(y_pred),
                "Price_Prediction_Bias": np.mean(y_pred - y_true),
            },
        }

        return business_report
