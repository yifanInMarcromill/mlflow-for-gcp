import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os

# Create a synthetic regression dataset
X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
def create_model():
    model = Sequential([
        Dense(64, activation='LeakyReLU', input_shape=(X_train.shape[1],)),
        Dense(64, activation='LeakyReLU'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def create_modelLeaky(alpha):
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],)),
        LeakyReLU(alpha=alpha),
        Dense(64),
        LeakyReLU(alpha=alpha),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

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
    
    # Create the model
    # model = create_model()
    model = create_modelLeaky(0.1)
    
    # Log parameters
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss", "mean_squared_error")
    mlflow.log_param("epochs", 200)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)
    
    # Log metrics
    for epoch in range(200):
        mlflow.log_metric("mse", history.history['loss'][epoch], step=epoch + 1)
        mlflow.log_metric("val_mse", history.history['val_loss'][epoch], step=epoch + 1)
    
    # Model training complete
    print("Model training complete")

    # Predict the test set
    y_pred = model.predict(X_test).flatten()

    # Binarize the predictions for classification metrics calculation
    threshold = np.median(y_pred)
    y_pred_binary = (y_pred > threshold).astype(int)
    y_test_binary = (y_test > threshold).astype(int)

    # Calculate classification metrics
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()

    # Log classification metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("true_positive", tp)
    mlflow.log_metric("true_negative", tn)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)

    # Log and register the model
    mlflow.keras.log_model(
        model=model,
        artifact_path="test",
        registered_model_name="2-Layer-MLP",
    )

    # Output run ID and model URI for further use
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/test"
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

    # Load the model back and make predictions to verify
    loaded_model = mlflow.keras.load_model(model_uri)
    sample_input = X_test[:5]
    sample_predictions = loaded_model.predict(sample_input)
    print(f"Sample Predictions: {sample_predictions}")
