import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


class MLExperimentFramework:
    """
    Comprehensive ML experimentation framework with multiple algorithms
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_params = {}
        self.feature_importance = {}

    def setup_models(self):
        """Initialize different ML algorithms"""
        self.models = {
            "random_forest": RandomForestRegressor(random_state=42),
            "xgboost": xgb.XGBRegressor(random_state=42),
            "lightgbm": lgb.LGBMRegressor(random_state=42, verbose=-1),
            # 'catboost': cb.CatBoostRegressor(random_state=42, verbose=False),
            # 'gradient_boosting': GradientBoostingRegressor(random_state=42),
            # 'elastic_net': ElasticNet(random_state=42),
            # 'neural_network': MLPRegressor(random_state=42, max_iter=500)
        }

    def get_hyperparameter_space(self, model_name):
        """Define hyperparameter search spaces for each model"""
        spaces = {
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
            "lightgbm": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 50, 100],
                "feature_fraction": [0.8, 0.9, 1.0],
            },
            "catboost": {
                "iterations": [100, 200, 300],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.1, 0.2],
                "l2_leaf_reg": [1, 3, 5],
            },
        }
        return spaces.get(model_name, {})

    def hyperparameter_tuning_optuna(
        self, model_name, X_train, y_train, X_val, y_val, n_trials=1
    ):
        """Hyperparameter tuning using Optuna"""

        def objective(trial):
            if model_name == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3
                    ),
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.7, 1.0
                    ),
                    "random_state": 42,
                }
                model = xgb.XGBRegressor(**params)

            elif model_name == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "feature_fraction": trial.suggest_float(
                        "feature_fraction", 0.7, 1.0
                    ),
                    "random_state": 42,
                    "verbose": -1,
                }
                model = lgb.LGBMRegressor(**params)

            elif model_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 2, 10
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 1, 5
                    ),
                    "random_state": 42,
                }
                model = RandomForestRegressor(**params)

            else:
                return float("inf")

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_pred)

            return mape

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params[model_name] = study.best_params
        return study.best_params

    def cross_validate_models(self, X, y, cv_folds=5):
        """Time series cross-validation for all models"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")

            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Fit model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Calculate MAPE
                mape = mean_absolute_percentage_error(y_val, y_pred)
                cv_scores.append(mape)

            self.results[model_name] = {
                "cv_scores": cv_scores,
                "mean_cv_score": np.mean(cv_scores),
                "std_cv_score": np.std(cv_scores),
            }

    def train_best_models(self, X_train, y_train, X_val, y_val):
        """Train models with best hyperparameters"""
        self.setup_models()

        # Key models to optimize
        key_models = ["xgboost", "lightgbm", "random_forest"]

        for model_name in key_models:
            print(f"Tuning hyperparameters for {model_name}...")
            best_params = self.hyperparameter_tuning_optuna(
                model_name, X_train, y_train, X_val, y_val
            )

            # Train with best parameters
            if model_name == "xgboost":
                model = xgb.XGBRegressor(**best_params)
            elif model_name == "lightgbm":
                model = lgb.LGBMRegressor(**best_params)
            elif model_name == "random_forest":
                model = RandomForestRegressor(**best_params)

            model.fit(X_train, y_train)
            self.models[model_name] = model

            # Store feature importance
            if hasattr(model, "feature_importances_"):
                self.feature_importance[model_name] = (
                    model.feature_importances_
                )
