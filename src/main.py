import os
import warnings

import dotenv
import numpy as np
import pandas as pd

from data.pre_processing import DataPreprocessor
from pipeline.dynamic_pricing_pipeline import DynamicPricingPipeline

dotenv.load_dotenv()

warnings.filterwarnings("ignore")


def create_sample_data(n_samples=1000):
    """Create synthetic sample data compatible with AdvancedPricingFeatureEngineer"""
    np.random.seed(42)

    # Constants
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")
    brands = ["Nirma", "Surf Excel"]
    fc_ids = ["FC_01", "FC_02", "FC_03", "FC_04", "FC_05"]

    data = []
    for i in range(n_samples):
        date = dates[i]
        brand = np.random.choice(brands)
        fc_id = np.random.choice(fc_ids)

        base_price = np.random.uniform(20, 80)
        mrp = base_price * np.random.uniform(1.2, 1.8)
        discount_rate = np.random.uniform(0.1, 0.4)
        selling_price = mrp * (1 - discount_rate)

        demand = int(
            np.random.poisson(20) + 30 * (1 / (selling_price + 1) * 100)
        )
        units_sold = min(demand, np.random.poisson(25))

        data.append(
            {
                "Date": date,
                "Brand": brand,
                "FC_ID": fc_id,
                # Pricing related
                "BasePrice": base_price,
                "MRP_x": mrp,
                "NoPromoPrice": mrp * 0.95,
                "MRP_y": mrp * np.random.uniform(0.98, 1.02),
                "DiscountRate": discount_rate,
                "SellingPrice": selling_price,
                "FinalPrice": selling_price * np.random.uniform(0.98, 1.02),
                # Customer behavior
                "CTR": np.random.uniform(0.02, 0.15),
                "AbandonedCartRate": np.random.uniform(0.3, 0.7),
                "BounceRate": np.random.uniform(0.2, 0.6),
                "FunnelDrop_ViewToCart": np.random.uniform(0.1, 0.4),
                "FunnelDrop_CartToCheckout": np.random.uniform(0.1, 0.3),
                "ReturningVisitorRatio": np.random.uniform(0.2, 0.8),
                "AvgSessionDuration_sec": np.random.uniform(60, 600),
                # Inventory & demand
                "StockStart": np.random.randint(50, 200),
                "Demand": demand,
                "UnitsSold": units_sold,
                "DemandFulfilled": min(demand, units_sold),
                "Backorders": max(0, demand - units_sold),
                "StockEnd": np.random.randint(10, 150),
                "ReorderPoint": np.random.randint(20, 50),
                "OrderPlaced": np.random.choice([0, 1]),
                "OrderQty": np.random.randint(50, 200),
                "LeadTimeFloat": np.random.uniform(1, 14),
                "SafetyStock": np.random.randint(10, 40),
                # Market indicator
                "IsMetro": np.random.choice([0, 1]),
            }
        )

    df = pd.DataFrame(data)
    df.reset_index(drop=True, inplace=True)
    return df


def run_pricing_pipeline():
    blob_conn_str = os.getenv("BLOB_CONNECTION_STRING")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(blob_conn_str)

    print("Preprocessing Data...")
    df = preprocessor.run_complete_preprocessing()
    print("Preprocessing Data - Done")

    print("Initializing pipeline...")
    pipeline = DynamicPricingPipeline("pricing_optimization_demo")

    print("Running complete pipeline...")
    results = pipeline.run_complete_pipeline(df, target_column="SellingPrice")

    if results:
        print("\nPipeline completed successfully!")
        print(f"Best model: {results['best_model_name']}")
        print(f"Number of features used: {len(results['feature_columns'])}")

        # Example of making predictions on new data
        print("\nTesting predictions on new data...")
        new_sample = df.tail(60).copy()
        predictions, *_ = pipeline.predict_prices(new_sample.tail(60))
        predictions = predictions.flatten()  # Safe for (n, 1) shape

        print("Sample predictions:")
        for i, (actual, pred) in enumerate(
            zip(new_sample["SellingPrice"].values, predictions)
        ):
            print(
                f"  Sample {i+1}: Actual=${float(actual):.2f}, Predicted=${float(pred):.2f}"
            )

    return pipeline, results


# ==================== ADVANCED UTILITIES ====================


class PricingOptimizationRecommendationEngine:
    """
    Advanced recommendation engine for pricing optimization
    """

    def __init__(self, trained_pipeline):
        self.pipeline = trained_pipeline
        self.price_elasticity_model = None

    def calculate_price_elasticity(
        self, data, price_changes=[-0.1, -0.05, 0, 0.05, 0.1]
    ):
        """Calculate price elasticity for different scenarios"""
        elasticity_results = {}

        for price_change in price_changes:
            modified_data = data.copy()
            modified_data["SellingPrice"] = modified_data["SellingPrice"] * (
                1 + price_change
            )

            # Re-engineer features with new prices
            modified_processed = self.pipeline.feature_engineer.transform(
                modified_data
            )

            # Predict demand with new prices
            X_modified, _ = self.pipeline.feature_engineer.transform(
                modified_data
            )
            X_modified = X_modified[self.pipeline.feature_columns].fillna(0)
            predicted_demand = self.pipeline.best_model.predict(X_modified)

            elasticity_results[price_change] = {
                "price_change_pct": price_change * 100,
                "predicted_prices": modified_data["SellingPrice"].values,
                "predicted_demand": predicted_demand,
                "revenue_impact": np.sum(
                    predicted_demand * modified_data["SellingPrice"]
                ),
            }

        return elasticity_results

    def recommend_optimal_prices(self, data, profit_margin=0.3):
        """Recommend optimal prices based on profit maximization"""
        # Calculate elasticity for different price points
        elasticity_results = self.calculate_price_elasticity(data)

        recommendations = []
        for idx, row in data.iterrows():
            current_price = row["SellingPrice"]
            cost = current_price * (1 - profit_margin)  # Estimated cost

            best_profit = 0
            best_price = current_price

            for price_change, results in elasticity_results.items():
                new_price = (
                    results["predicted_prices"][idx]
                    if idx < len(results["predicted_prices"])
                    else current_price
                )
                predicted_units = (
                    results["predicted_demand"][idx]
                    if idx < len(results["predicted_demand"])
                    else 0
                )

                profit = (new_price - cost) * predicted_units

                if profit > best_profit:
                    best_profit = profit
                    best_price = new_price

            recommendations.append(
                {
                    "current_price": current_price,
                    "recommended_price": best_price,
                    "expected_profit_increase": best_profit,
                    "price_change_pct": (best_price - current_price)
                    / current_price
                    * 100,
                }
            )

        return recommendations


if __name__ == "__main__":
    # Example usage
    print("Dynamic Pricing ML Pipeline - Complete Implementation")
    print("=" * 60)

    # Run the example pipeline
    pipeline, results = run_pricing_pipeline()

    if results:
        # Initialize recommendation engine
        sample_data = create_sample_data(20)
        recommender = PricingOptimizationRecommendationEngine(pipeline)
        recommendations = recommender.recommend_optimal_prices(sample_data)

        print(f"\nGenerated {len(recommendations)} pricing recommendations")
        print("Sample recommendations:")
        for i, rec in enumerate(recommendations[:5]):
            print(
                f"  Product {i+1}: ${rec['current_price']:.2f} → ${rec['recommended_price']:.2f} "
                f"({rec['price_change_pct']:.1f}% change)"
            )
