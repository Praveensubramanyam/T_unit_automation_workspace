import os
import pickle
import tempfile

import joblib
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient


class MLflowPricingTracker:
    """
    MLflow integration for experiment tracking and model management
    """

    def __init__(self, experiment_name="dynamic_pricing_optimization"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.setup_experiment()

    def setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"MLflow setup warning: {e}")

    def log_feature_engineering_run(self, feature_stats, feature_names):
        """Log feature engineering experiment"""
        with mlflow.start_run(run_name="feature_engineering"):
            # Log feature statistics
            mlflow.log_param("num_features", len(feature_names))
            mlflow.log_param(
                "feature_names", str(feature_names[:10])
            )  # Log first 10

            # Log feature statistics as metrics
            for stat_name, stat_value in feature_stats.items():
                if isinstance(stat_value, (int, float)):
                    mlflow.log_metric(f"feature_{stat_name}", stat_value)

    def log_model_experiment(
        self,
        model,
        model_name,
        params,
        metrics,
        feature_importance=None,
        model_artifacts=None,
    ):
        """Log complete model experiment with robust error handling"""
        with mlflow.start_run(run_name=f"{model_name}_experiment"):
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            mlflow.log_metric(
                                f"{metric_name}_{sub_metric}", sub_value
                            )
                elif isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            # Log model with fallback approach
            self._log_model_with_fallback(model, model_name)

            # Log feature importance
            if feature_importance is not None:
                importance_dict = {
                    f"feature_{i}": imp
                    for i, imp in enumerate(feature_importance)
                }
                mlflow.log_metrics(importance_dict)

            # Log additional artifacts
            if model_artifacts:
                for artifact_name, artifact_path in model_artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)

    def _log_model_with_fallback(self, model, model_name):
        """Log model with multiple fallback strategies"""
        try:
            # Primary approach: Use specific model flavor
            if model_name == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"{model_name}_dynamic_pricing",
                )
            elif model_name == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"{model_name}_dynamic_pricing",
                )
            elif model_name == "catboost":
                mlflow.catboost.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"{model_name}_dynamic_pricing",
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"{model_name}_dynamic_pricing",
                )
            print(
                f"✓ Successfully logged {model_name} model using specific flavor"
            )

        except Exception as e:
            print(f"⚠ Primary model logging failed for {model_name}: {e}")
            self._fallback_model_logging(model, model_name)

    def _fallback_model_logging(self, model, model_name):
        """Fallback model logging using basic sklearn or pickle"""
        try:
            # Fallback 1: Try basic sklearn logging without registration
            if hasattr(model, "predict"):
                mlflow.sklearn.log_model(model, artifact_path="model")
                print(
                    f"✓ Fallback 1: Logged {model_name} model using sklearn flavor"
                )
                return
        except Exception as e:
            print(f"⚠ Sklearn fallback failed: {e}")

        try:
            # Fallback 2: Save as pickle and log as artifact
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(model, f)
                temp_path = f.name

            mlflow.log_artifact(temp_path, "model")
            os.unlink(temp_path)
            print(
                f"✓ Fallback 2: Logged {model_name} model as pickle artifact"
            )

        except Exception as e:
            print(f"⚠ Pickle fallback also failed: {e}")

        try:
            # Fallback 3: Use joblib
            with tempfile.NamedTemporaryFile(
                suffix=".joblib", delete=False
            ) as f:
                joblib.dump(model, f.name)
                temp_path = f.name

            mlflow.log_artifact(temp_path, "model")
            os.unlink(temp_path)
            print(f"✓ Fallback 3: Logged {model_name} model using joblib")

        except Exception as e:
            print(f"✗ All model logging methods failed for {model_name}: {e}")

    def compare_model_runs(self, experiment_name=None):
        """Compare model runs and return best performing model"""
        if experiment_name is None:
            experiment_name = self.experiment_name

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if len(runs) == 0:
            return None

        # Sort by MAPE (lower is better)
        runs_sorted = runs.sort_values("metrics.MAPE", ascending=True)
        best_run = runs_sorted.iloc[0]

        return {
            "best_run_id": best_run["run_id"],
            "best_model_name": best_run.get("tags.mlflow.runName", "unknown"),
            "best_mape": best_run["metrics.MAPE"],
            "best_r2": best_run.get("metrics.R2", 0),
            "run_details": best_run,
        }

    def load_best_model(self, run_id, fallback_to_artifact=True):
        """Load the best model from MLflow with fallback options"""
        try:
            # Try loading as MLflow model first
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            print("✓ Loaded model using MLflow pyfunc")
            return model

        except Exception as e:
            print(f"⚠ MLflow model loading failed: {e}")

            if fallback_to_artifact:
                return self._load_model_from_artifacts(run_id)
            else:
                raise e

    def _load_model_from_artifacts(self, run_id):
        """Load model from artifacts if MLflow model loading fails"""
        try:
            # Download model artifacts
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model"
            )

            # Try to load from different formats
            model_files = os.listdir(artifact_path)

            for file_name in model_files:
                file_path = os.path.join(artifact_path, file_name)

                if file_name.endswith(".pkl"):
                    with open(file_path, "rb") as f:
                        model = pickle.load(f)
                    print("✓ Loaded model from pickle artifact")
                    return model

                elif file_name.endswith(".joblib"):
                    model = joblib.load(file_path)
                    print("✓ Loaded model from joblib artifact")
                    return model

            raise Exception("No compatible model files found in artifacts")

        except Exception as e:
            print(f"✗ Artifact loading also failed: {e}")
            raise e
