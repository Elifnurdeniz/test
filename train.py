import mlflow

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://34.28.175.237:5000")  # Replace with your actual MLflow server URL

# Set an experiment (creates it if it doesn't exist)
mlflow.set_experiment("MLflow Connectivity Test")

# Start an MLflow run
with mlflow.start_run():
    mlflow.log_param("test_param", 1)
    mlflow.log_metric("test_metric", 0.99)
    print("MLflow logging successful!")