# Imports librairies
import os
import sys
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Apple_Models")

run_name = "first_run"
artifact_path = "rf_apples"

# Import Database
data = pd.read_csv("fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype("float")
y = data["demand"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameters
n_estimators = int(sys.argv[1])
max_depth = int(sys.argv[2])

params = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "random_state": 42,
}
rf = RandomForestRegressor(**params)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

# Store information in tracking server
with mlflow.start_run(run_name=run_name):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    logged_model = mlflow.sklearn.log_model(
        sk_model=rf,
        input_example=X_val,
        artifact_path=artifact_path,
    )
    print(f"Logged model at {logged_model.model_uri}")
