import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Create a synthetic regression dataset
X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
params = {"max_iter": 1, "tol": None, "random_state": 42, "warm_start": True}
model = SGDRegressor(**params)

# Set environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = "dea-intern-202404"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "Arizona123!"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("https://dae-intern-mlflow-dsu2uxvhnq-uc.a.run.app")
mlflow.set_experiment("DAE")

# Start an MLflow run
with mlflow.start_run() as run:
    print("Started MLflow run")
    
    # Log parameters
    mlflow.log_params(params)

    # Train the model in epochs and log MSE at each epoch
    epochs = 50
    for epoch in range(epochs):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Epoch {epoch + 1}, Mean Squared Error: {mse}")
        mlflow.log_metric("mse", mse, step=epoch + 1)
    
    # Model training complete
    print("Model training complete")

    # Infer the model signature
    signature = infer_signature(X_test, y_pred)

    # Log and register the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="test",
        signature=signature,
        registered_model_name="try1",
    )

    # Output run ID and model URI for further use
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/test"
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

    # Load the model back and make predictions to verify
    loaded_model = mlflow.sklearn.load_model(model_uri)
    sample_input = X_test[:5]
    sample_predictions = loaded_model.predict(sample_input)
    print(f"Sample Predictions: {sample_predictions}")
