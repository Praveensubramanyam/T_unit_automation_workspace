# train.py

from azureml.core import Run
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Start Azure ML run
run = Run.get_context()

# Load data from mounted input
input_path = os.path.join("inputs", "cleaned_data")
df = pd.read_parquet(input_path)

# Prepare features
X = df.drop(columns=["TransactionDate", "FinalPrice"])
y = df["FinalPrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict & evaluate
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)

# Log model and metrics
mlflow.log_metric("rmse", rmse)
mlflow.sklearn.log_model(model, "rf-model")

run.complete()
