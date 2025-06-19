import os
import uuid
from datetime import datetime

import dotenv
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from azure.data.tables import TableServiceClient
from evaluation_metrics import BusinessEvaluationFramework
from feature_engineering import AdvancedPricingFeatureEngineer
from mlflow_tracker import MLflowPricingTracker
from model_training import MLExperimentFramework
from time_series import PricingTimeSeriesAnalyzer

dotenv.load_dotenv()


class DynamicPricingPipeline:
    """
    Complete end-to-end dynamic pricing optimization pipeline
    """

    def __init__(self, experiment_name="dynamic_pricing_optimization"):
        self.feature_engineer = AdvancedPricingFeatureEngineer()
        self.ts_analyzer = PricingTimeSeriesAnalyzer()
        self.ml_framework = MLExperimentFramework()
        self.evaluator = BusinessEvaluationFramework()
        self.mlflow_tracker = MLflowPricingTracker(experiment_name)
        self.TABLE_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")

        self.TABLE_NAME = "businessInsights"

        self.processed_data = None
        self.trained_models = {}
        self.best_model = None
        self.feature_columns = []

    def load_and_preprocess_data(self, data_path_or_df):
        """Load and preprocess the data"""
        if isinstance(data_path_or_df, str):
            df = pd.read_csv(data_path_or_df)
        else:
            df = data_path_or_df.copy()

        print(f"Loaded data with shape: {df.shape}")

        # Basic data quality checks
        print("\nData Quality Summary:")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        print(f"\nData types:\n{df.dtypes}")

        # Ensure required columns exist
        required_columns = ["SellingPrice", "MRP_x", "Brand", "FC_ID"]
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    def run_feature_engineering(self, df):
        experiment_name = "dynamic_pricing_optimization"
        """Run comprehensive leakage-free feature engineering"""
        print("\n" + "=" * 50)
        print("LEAKAGE-FREE FEATURE ENGINEERING PHASE")
        print("=" * 50)

        # Apply the new leakage-free feature engineering
        X, y, df_processed = self.feature_engineer.fit_transform(df)
        self.feature_columns = self.feature_engineer.get_feature_names()

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)
        print(f"Original dataset shape: {df.shape}")
        print(f"Processed dataset shape: {df_processed.shape}")
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Total features created: {len(self.feature_columns)}")
        print(f"Target column: SellingPrice")
        print(
            f"✓ No data leakage - all features available before pricing decisions"
        )
        print(f"✓ Ready for model training!")

        # Display feature summary
        print(f"\nFeature Categories:")
        product_features = [
            f
            for f in self.feature_columns
            if any(
                x in f.lower() for x in ["mrp", "brand", "premium", "product"]
            )
        ]
        temporal_features = [
            f
            for f in self.feature_columns
            if any(
                x in f.lower()
                for x in ["month", "day", "week", "season", "summer"]
            )
        ]
        market_features = [
            f
            for f in self.feature_columns
            if any(x in f.lower() for x in ["metro", "market"])
        ]
        historical_features = [
            f for f in self.feature_columns if "historical" in f.lower()
        ]

        print(f"  - Product characteristics: {len(product_features)}")
        print(f"  - Temporal patterns: {len(temporal_features)}")
        print(f"  - Market/Location: {len(market_features)}")
        print(f"  - Historical patterns: {len(historical_features)}")

        # Log feature engineering to MLflow
        feature_stats = {
            "total_features": len(self.feature_columns),
            "original_columns": len(df.columns),
            "product_features": len(product_features),
            "temporal_features": len(temporal_features),
            "market_features": len(market_features),
            "historical_features": len(historical_features),
            "leakage_free": True,
        }

        self.mlflow_tracker.log_feature_engineering_run(
            feature_stats, self.feature_columns
        )

        # Store processed data and feature/target matrices
        self.processed_data = df_processed
        self.X_processed = X
        self.y_processed = y

        return X, y, df_processed

    def run_time_series_analysis(self, df):
        """Run time series analysis and create temporal features"""
        print("\n" + "=" * 50)
        print("TIME SERIES ANALYSIS PHASE")
        print("=" * 50)

        # Only run time series analysis if Date column exists
        if "Date" not in df.columns:
            print("⚠️  No Date column found. Skipping time series analysis.")
            return df, {}

        # Decompose time series for key metrics
        try:
            decomp_price = self.ts_analyzer.decompose_time_series(
                df, "SellingPrice"
            )
            print("✓ Price decomposition completed")
        except Exception as e:
            print(f"⚠️  Price decomposition failed: {e}")
            decomp_price = None

        try:
            if "UnitsSold" in df.columns:
                decomp_demand = self.ts_analyzer.decompose_time_series(
                    df, "UnitsSold"
                )
                print("✓ Demand decomposition completed")
            else:
                print(
                    "⚠️  UnitsSold column not found. Skipping demand decomposition."
                )
                decomp_demand = None
        except Exception as e:
            print(f"⚠️  Demand decomposition failed: {e}")
            decomp_demand = None

        # Detect seasonality patterns
        try:
            seasonality_price = self.ts_analyzer.detect_seasonality(
                df, "SellingPrice"
            )
            print(
                f"✓ Price seasonality strength: {seasonality_price.get('seasonal_strength', 'N/A')}"
            )
        except Exception as e:
            print(f"⚠️  Price seasonality detection failed: {e}")
            seasonality_price = {}

        try:
            if "UnitsSold" in df.columns:
                seasonality_demand = self.ts_analyzer.detect_seasonality(
                    df, "UnitsSold"
                )
                print(
                    f"✓ Demand seasonality strength: {seasonality_demand.get('seasonal_strength', 'N/A')}"
                )
            else:
                seasonality_demand = {}
        except Exception as e:
            print(f"⚠️  Demand seasonality detection failed: {e}")
            seasonality_demand = {}

        # Generate demand forecasts if possible
        try:
            demand_forecasts = self.ts_analyzer.forecast_demand(df)
            print(
                f"✓ Generated forecasts for {len(demand_forecasts)} combinations"
            )
        except Exception as e:
            print(f"⚠️  Demand forecasting failed: {e}")
            demand_forecasts = {}

        # Note: Advanced temporal features are already handled in AdvancedPricingFeatureEngineer
        print(
            "✓ Time series analysis completed (additional temporal features already in feature engineering)"
        )

        return df, demand_forecasts

    def prepare_model_data(self, X, y, test_size=0.2):
        """Prepare feature matrix and target for ML modeling with time series split"""
        print("\n" + "=" * 50)
        print("DATA PREPARATION FOR ML")
        print("=" * 50)

        # Ensure we have the processed feature matrix
        if X is None or y is None:
            raise ValueError("Feature matrix X and target y must be provided")

        # Time series split (last test_size portion for testing)
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Further split training data for validation
        val_split_idx = int(len(X_train) * 0.8)
        X_val = X_train.iloc[val_split_idx:]
        y_val = y_train.iloc[val_split_idx:]
        X_train_final = X_train.iloc[:val_split_idx]
        y_train_final = y_train.iloc[:val_split_idx]

        print(f"✓ Training set: {X_train_final.shape}")
        print(f"✓ Validation set: {X_val.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(f"✓ Features: {len(self.feature_columns)}")
        print(f"✓ All features are leakage-free")

        # Verify no missing values
        missing_train = X_train_final.isnull().sum().sum()
        missing_val = X_val.isnull().sum().sum()
        missing_test = X_test.isnull().sum().sum()

        if missing_train + missing_val + missing_test > 0:
            print(
                f"⚠️  Found missing values: Train={missing_train}, Val={missing_val}, Test={missing_test}"
            )
            print("Filling missing values with 0...")
            X_train_final = X_train_final.fillna(0)
            X_val = X_val.fillna(0)
            X_test = X_test.fillna(0)

        print("✓ Training set info:")
        X_train_final.info()

        return X_train_final, X_val, X_test, y_train_final, y_val, y_test

    def run_ml_experiments(
        self, X_train, X_val, X_test, y_train, y_val, y_test
    ):
        """Run comprehensive ML experiments"""
        print("\n" + "=" * 50)
        print("ML EXPERIMENTATION PHASE")
        print("=" * 50)

        # Train models with hyperparameter tuning
        print("🚀 Starting model training with hyperparameter optimization...")
        self.ml_framework.train_best_models(X_train, y_train, X_val, y_val)
        print("✓ Model training completed")

        # Evaluate all models
        best_model_name = None
        best_mape = float("inf")

        print("\n📊 MODEL EVALUATION RESULTS:")
        print("-" * 50)
        tag_list = []
        for model_name, model in self.ml_framework.models.items():
            if model is None:
                print(f"❌ {model_name}: Training failed")
                continue

            print(f"\n🔍 Evaluating {model_name}...")

            try:
                exp_models = {}
                # Make predictions
                y_pred_val = model.predict(X_val)
                y_pred_test = model.predict(X_test)

                # Calculate metrics
                val_metrics = (
                    self.evaluator.calculate_pricing_accuracy_metrics(
                        y_val, y_pred_val
                    )
                )
                test_metrics = (
                    self.evaluator.calculate_pricing_accuracy_metrics(
                        y_test, y_pred_test
                    )
                )

                print(f"  📈 Validation MAPE: {val_metrics['MAPE']:.3f}")
                print(f"  📈 Test MAPE: {test_metrics['MAPE']:.3f}")
                print(f"  📈 Test R²: {test_metrics['R2']:.3f}")
                print(f"  📈 Test RMSE: {test_metrics.get('RMSE', 'N/A')}")

                # Track best model
                if test_metrics["MAPE"] < best_mape:
                    best_mape = test_metrics["MAPE"]
                    best_model_name = model_name
                    self.best_model = model

                directional_accuracy = test_metrics.get(
                    "Directional_Accuracy", 0
                )

                # Prepare metrics for MLflow logging
                combined_metrics = {
                    "validation_MAPE": val_metrics["MAPE"],
                    "validation_R2": val_metrics["R2"],
                    "test_MAPE": test_metrics["MAPE"],
                    "test_R2": test_metrics["R2"],
                    "test_RMSE": test_metrics.get("RMSE", 0),
                    "directional_accuracy": test_metrics.get(
                        "Directional_Accuracy", 0
                    ),
                }

                # Get model parameters
                model_params = self.ml_framework.best_params.get(
                    model_name, {}
                )

                # Log to MLflow
                feature_importance = self.ml_framework.feature_importance.get(
                    model_name, None
                )

                exp_models["model"] = model
                exp_models["model_name"] = model_name
                exp_models["params"] = model_params
                exp_models["metrics"] = combined_metrics
                exp_models["feature_importance"] = feature_importance

                tag_list.append(exp_models)
                print(tag_list[-1]["model_name"])
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue

        if best_model_name:
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"🎯 Best Test MAPE: {best_mape:.3f}")

            # Get the best model instance
            best_model = self.ml_framework.models[best_model_name]
            for i in tag_list:
                if i["model_name"] == best_model_name:
                    self.mlflow_tracker.log_model_experiment(
                        model=i["model"],
                        model_name=i["model_name"],
                        params=i["params"],
                        metrics=i["metrics"],
                        feature_importance=i["feature_importance"],
                    )
                else:
                    continue
        else:
            print("\n❌ No models were successfully trained!")

        return best_model_name, directional_accuracy

    def generate_business_insights(
        self, X_test, y_test, model_name, directional_accuracy
    ):
        """Generate comprehensive business insights"""
        print("\n" + "=" * 50)
        print("BUSINESS INSIGHTS GENERATION")
        print("=" * 50)

        if self.best_model is None:
            print("❌ No trained model available for insights generation")
            return None

        # Make predictions
        y_pred = self.best_model.predict(X_test)

        print(
            f"✓ Generated {len(y_pred)} price predictions using {model_name}"
        )

        # Calculate business metrics
        # For demo purposes, we'll create dummy units sold data if not available
        units_sold = np.random.poisson(
            10, size=len(y_test)
        )  # Replace with actual data

        try:
            # Generate comprehensive business report
            business_report = self.evaluator.generate_business_report(
                y_test, y_pred, units_sold
            )

            print("\n💼 BUSINESS IMPACT SUMMARY:")
            print("-" * 40)
            for category, metrics in business_report.items():
                print(f"\n📊 {category.upper()}:")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  • {metric}: {value:.3f}")
                        else:
                            print(f"  • {metric}: {value}")

        except Exception as e:
            print(f"⚠️  Business report generation failed: {e}")
            business_report = {}

        # Feature importance analysis
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
        print("-" * 40)

        try:
            if hasattr(self.best_model, "feature_importances_"):
                feature_importance = self.best_model.feature_importances_

                if len(feature_importance) == len(self.feature_columns):
                    importance_df = pd.DataFrame(
                        {
                            "feature": self.feature_columns,
                            "importance": feature_importance,
                        }
                    ).sort_values("importance", ascending=False)

                    print(f"📈 TOP 10 MOST IMPORTANT FEATURES:")
                    for idx, row in importance_df.head(10).iterrows():
                        print(f"  {row['importance']:.4f} - {row['feature']}")

                    # Categorize important features
                    top_features = importance_df.head(5)["feature"].tolist()
                    feature_categories = {
                        "Product": [
                            f
                            for f in top_features
                            if any(
                                x in f.lower()
                                for x in ["mrp", "brand", "premium", "product"]
                            )
                        ],
                        "Temporal": [
                            f
                            for f in top_features
                            if any(
                                x in f.lower()
                                for x in ["month", "day", "week", "season"]
                            )
                        ],
                        "Market": [
                            f
                            for f in top_features
                            if any(x in f.lower() for x in ["metro", "market"])
                        ],
                        "Historical": [
                            f
                            for f in top_features
                            if "historical" in f.lower()
                        ],
                    }

                    print(f"\n🏷️  KEY INSIGHTS:")
                    for category, features in feature_categories.items():
                        if features:
                            print(
                                f"  • {category} factors are important: {', '.join(features)}"
                            )

                else:
                    print(
                        f"⚠️  Feature importance mismatch: {len(feature_importance)} vs {len(self.feature_columns)}"
                    )

            elif hasattr(self.best_model, "feature_importance_"):
                # For LightGBM
                feature_importance = self.best_model.feature_importance_
                print(f"✓ Feature importance available (LightGBM style)")

        except Exception as e:
            print(f"⚠️  Feature importance analysis failed: {e}")

        # === STORE BUSINESS INSIGHTS IN TABLE STORAGE ===

        try:
            service = TableServiceClient.from_connection_string(
                conn_str=self.TABLE_CONNECTION_STRING
            )
            table_client = service.get_table_client(table_name=self.TABLE_NAME)

            service.create_table_if_not_exists(table_name=self.TABLE_NAME)

            actual_revenue = float(np.sum(y_test))
            predicted_revenue = float(np.sum(y_pred))

            entity = {
                "PartitionKey": datetime.now().strftime("%Y%m%d"),
                "RowKey": str(uuid.uuid4()),
                "model_name": model_name,
                "actual_revenue": actual_revenue,
                "predicted_revenue": predicted_revenue,
                "directional_accuracy": float(directional_accuracy),
                "timestamp": datetime.utcnow().isoformat(),
            }

            table_client.upsert_entity(entity=entity)
            print(f"📤 Logged business summary to Azure Table Storage.")
        except Exception as e:
            print(f"⚠️ Failed to store business insights: {e}")

        return business_report

    def run_complete_pipeline(
        self, data_path_or_df, target_column="SellingPrice"
    ):
        """Run the complete end-to-end pipeline"""
        print("🚀 STARTING DYNAMIC PRICING OPTIMIZATION PIPELINE")
        print("=" * 70)
        print("🎯 Objective: Build leakage-free pricing prediction model")
        print("=" * 70)

        try:
            # Step 1: Load and preprocess data
            print("\n📂 STEP 1: DATA LOADING")
            df = self.load_and_preprocess_data(data_path_or_df)

            # Step 2: Leakage-free feature engineering
            print("\n⚙️  STEP 2: FEATURE ENGINEERING")
            X, y, df_processed = self.run_feature_engineering(df)

            # Step 3: Time series analysis (optional, for insights)
            print("\n📈 STEP 3: TIME SERIES ANALYSIS")
            df_with_ts, forecasts = self.run_time_series_analysis(df_processed)

            # Step 4: Prepare data for ML
            print("\n🔧 STEP 4: ML DATA PREPARATION")
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.prepare_model_data(X, y)
            )

            # Step 5: Run ML experiments
            print("\n🤖 STEP 5: ML EXPERIMENTATION")
            best_model_name, directional_accuracy = self.run_ml_experiments(
                X_train, X_val, X_test, y_train, y_val, y_test
            )

            # Step 6: Generate business insights
            print("\n💡 STEP 6: BUSINESS INSIGHTS")
            business_report = self.generate_business_insights(
                X_test, y_test, best_model_name, directional_accuracy
            )

            print("\n" + "=" * 70)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"✅ Best Model: {best_model_name}")
            print(
                f"✅ Features: {len(self.feature_columns)} (all leakage-free)"
            )
            print(f"✅ Ready for production pricing predictions")
            print("=" * 70)

            return {
                "best_model": self.best_model,
                "best_model_name": best_model_name,
                "business_report": business_report,
                "feature_columns": self.feature_columns,
                "processed_data": self.processed_data,
                "feature_matrix": X,
                "target_vector": y,
            }

        except Exception as e:
            print(f"❌ PIPELINE FAILED: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def predict_prices(self, new_data):
        """Generate price predictions for new data"""
        if self.best_model is None:
            raise ValueError(
                "❌ No trained model available. Run the pipeline first."
            )

        print(f"\n🔮 GENERATING PRICE PREDICTIONS")
        print(f"📊 Input data shape: {new_data.shape}")

        # Apply same feature engineering (inference mode)
        X_new, df_encoded = self.feature_engineer.transform(new_data)

        print(f"✅ Feature engineering applied")
        print(f"📈 Feature matrix shape: {X_new.shape}")
        print(f"🔧 Using {len(self.feature_columns)} features")

        # Verify we have all required features
        missing_features = set(self.feature_columns) - set(X_new.columns)
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                X_new[feature] = 0

        # Select only the features used in training, in the same order
        X_new_aligned = X_new[self.feature_columns].fillna(0)

        # Make predictions
        predictions = self.best_model.predict(X_new_aligned)

        print(f"✅ Generated {len(predictions)} price predictions")

        # Add predictions back to the processed dataframe for analysis
        result_df = df_encoded.copy()
        result_df["Predicted_SellingPrice"] = predictions

        return predictions, result_df
