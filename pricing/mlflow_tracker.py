import os
import pickle
import tempfile
from datetime import datetime

import joblib
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential
from mlflow.tracking import MlflowClient


class MLflowLifecycleManager:
    """
    Manages model lifecycle stages in MLflow for dynamic pricing models
    """

    def __init__(self, model_name="dynamic-pricing-tide"):
        self.model_name = model_name
        self.client = MlflowClient()
        self._ensure_model_exists()

    def create_model_endpoint(self, version):
        """Create or update online endpoint for a model version"""
        try:

            # Get workspace details
            ml_client = MLClient.from_config(
                credential=DefaultAzureCredential(), logging_enable=True
            )

            # Create unique endpoint name
            endpoint_name = f"{self.model_name}-endpoint".lower().replace(
                "_", "-"
            )

            # Define endpoint
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description=f"Endpoint for {self.model_name}",
                auth_mode="key",
            )

            try:
                model_ref = ml_client.models.get(
                    name=self.model_name, version=version
                )
                model_id = model_ref.id  # This will be the proper ARM ID
            except Exception as model_error:
                print(f"Warning: Could not get model reference: {model_error}")
                # Fallback to constructing the reference
                model_id = f"azureml:{self.model_name}:{version}"

            # Create or update endpoint
            endpoint_result = (
                ml_client.online_endpoints.begin_create_or_update(
                    endpoint
                ).result()
            )
            print(f"✓ Created/updated endpoint: {endpoint_name}")

            # Create deployment configuration
            deployment = ManagedOnlineDeployment(
                name=f"deployment-{version}",
                endpoint_name=endpoint_name,
                model=model_id,
                instance_type="Standard_DS2_v2",
                instance_count=1,
            )

            # Deploy model
            deployment_result = ml_client.begin_create_or_update(
                deployment
            ).result()
            print(f"✓ Deployed model version {version} to endpoint")

            # Add endpoint info to model version tags
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="endpoint_name",
                value=endpoint_name,
            )

            return endpoint_name

        except Exception as e:
            print(f"❌ Error creating endpoint: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _ensure_model_exists(self):
        """Ensure registered model exists"""
        try:
            model = self.client.get_registered_model(self.model_name)
        except Exception:
            self.client.create_registered_model(self.model_name)
            print(f"✓ Created new registered model: {self.model_name}")

    def clean_old_versions(self, keep_last_n=5):
        """Archive versions beyond keep_last_n for each stage"""
        try:
            versions = self.get_latest_versions()
            current_n = {
                stage: 0 for stage in ["Production", "Staging", "None"]
            }

            for version in versions:
                stage = version.current_stage
                current_n[stage] = current_n.get(stage, 0) + 1

                if current_n[stage] > keep_last_n and stage != "Production":
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=version.version,
                        stage="Archived",
                    )
                    print(
                        f"✓ Archived old version {version.version} from {stage}"
                    )

        except Exception as e:
            print(f"❌ Error cleaning old versions: {e}")

    def register_new_model(self, model, run_id, metrics, tags):
        """Register new model version with metadata"""
        try:
            # Add standard tags
            tags.update(
                {
                    "run_id": run_id,
                    "registration_timestamp": datetime.now().isoformat(),
                    "model_type": type(model).__name__,
                }
            )

            # Register model
            model_uri = f"runs:/{run_id}/model"
            model_details = mlflow.register_model(
                model_uri=model_uri, name=self.model_name
            )

            # Add tags and metrics
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_details.version,
                    key=key,
                    value=str(value),
                )

            print(f"✓ Registered new model version {model_details.version}")
            self.create_model_endpoint(model_details.version)
            return model_details.version

        except Exception as e:
            print(f"❌ Error registering new model: {e}")
            return None

    def get_latest_versions(self, stages=None):
        """Get latest versions for specified stages"""
        try:
            filter_string = f"name='{self.model_name}'"
            versions = self.client.search_model_versions(filter_string)
            if not versions:
                print(f"⚠️ No versions found for model {self.model_name}")
                return []
            if stages:
                versions = [v for v in versions if v.current_stage in stages]
            return sorted(versions, key=lambda x: int(x.version), reverse=True)
        except Exception as e:
            print(f"❌ Error getting model versions: {e}")
            return []

    def archive_existing_production(self):
        """Archive existing production models"""
        try:
            prod_versions = self.get_latest_versions(stages=["Production"])
            if not prod_versions:
                print("ℹ️ No production models to archive")
                return

            for version in prod_versions:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=version.version,
                    stage="Archived",
                )
                print(f"✓ Archived production version {version.version}")
        except Exception as e:
            print(f"❌ Error archiving production models: {e}")
            raise

    def promote_to_production(self, version, archive_existing=True):
        """Promote a model version to production"""
        try:
            # Verify version exists
            model_version = self.client.get_model_version(
                name=self.model_name, version=version
            )
            if not model_version:
                raise ValueError(f"Version {version} not found")

            if archive_existing:
                self.archive_existing_production()

            self.client.transition_model_version_stage(
                name=self.model_name, version=version, stage="Production"
            )
            print(f"✓ Promoted version {version} to Production")

            # Log transition event
            with mlflow.start_run(nested=True) as run:
                mlflow.log_params(
                    {
                        "promoted_version": version,
                        "previous_stage": model_version.current_stage,
                        "new_stage": "Production",
                        "transition_timestamp": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            print(f"❌ Error promoting model to production: {e}")
            raise


class MLflowPricingTracker:
    """
    MLflow integration for experiment tracking and model management
    """

    def __init__(self, experiment_name="dynamic_pricing_optimization"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.lifecycle_manager = MLflowLifecycleManager("dynamic-pricing-tide")
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
        with mlflow.start_run(run_name=f"{model_name}_experiment") as run:
            try:
                # Log parameters
                clean_params = {
                    k: (
                        float(v)
                        if isinstance(v, (np.float32, np.float64))
                        else v
                    )
                    for k, v in params.items()
                }
                mlflow.log_params(clean_params)

                # Log metrics
                clean_metrics = {}
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        for sub_metric, sub_value in metric_value.items():
                            if isinstance(sub_value, (int, float, np.number)):
                                clean_metrics[
                                    f"{metric_name}_{sub_metric}"
                                ] = float(sub_value)
                    elif isinstance(metric_value, (int, float, np.number)):
                        clean_metrics[metric_name] = float(metric_value)
                mlflow.log_metrics(clean_metrics)

                # Log model
                self._log_model_with_fallback(model, model_name)

                # Handle feature importance with numpy array checks
                if feature_importance is not None:
                    importance_array = np.array(feature_importance).ravel()
                    # Only log if any importance values are non-zero
                    if np.any(importance_array):
                        importance_dict = {
                            f"feature_{i}": float(imp)
                            for i, imp in enumerate(importance_array)
                            if not isinstance(imp, np.ndarray) or imp.size == 1
                        }
                        if importance_dict:
                            mlflow.log_metrics(importance_dict)

                tags = {
                    "experiment_name": self.experiment_name,
                    "data_version": str(params.get("data_version", "v1.0")),
                    "model_framework": model_name,
                    "feature_count": str(
                        len(feature_importance)
                        if np.any(feature_importance)
                        else 0
                    ),
                }

                version = self.lifecycle_manager.register_new_model(
                    model=model,
                    run_id=run.info.run_id,
                    metrics=clean_metrics,
                    tags=tags,
                )

                if (
                    version is not None
                ):  # Explicit None check instead of array comparison
                    self.lifecycle_manager.client.transition_model_version_stage(
                        name="dynamic-pricing-tide",
                        version=version,
                        stage="Staging",
                    )
                    self.lifecycle_manager.clean_old_versions()

            except Exception as e:
                print(f"❌ Error in model experiment logging: {str(e)}")

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

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",  # Only look at completed runs
        )

        if len(runs) == 0:
            return None

        metric_columns = runs.columns[runs.columns.str.contains("MAPE")]

        if len(metric_columns) == 0:
            print("⚠️ No MAPE metrics found in runs")
            return None

        # Try different possible MAPE column names
        if "metrics.test_MAPE" in metric_columns:
            mape_column = "metrics.test_MAPE"
        elif "metrics.MAPE" in metric_columns:
            mape_column = "metrics.MAPE"
        else:
            mape_column = metric_columns[0]

        # Filter valid MAPE values and sort
        valid_runs = runs[
            (runs[mape_column].notna())
            & (runs[mape_column] > 0.001)
            & (runs[mape_column] < 1.0)
            & (runs["status"] == "FINISHED")
        ]

        if len(valid_runs) == 0:
            print(
                "⚠️ No valid MAPE values found (all are zero, NaN, or infinite)"
            )
            return None

        runs_sorted = valid_runs.sort_values(mape_column, ascending=True)
        best_run = runs_sorted.iloc[0]

        best_mape = float(best_run[mape_column])

        if not isinstance(best_mape, (int, float)) or best_mape <= 0.001:
            print(f"⚠️ Invalid best MAPE value: {best_mape}")
            return None

        return {
            "best_run_id": best_run["run_id"],
            "best_model_name": best_run.get("tags.mlflow.runName", "unknown"),
            "best_mape": float(best_mape),
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
