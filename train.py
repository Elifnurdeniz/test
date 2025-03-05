import mlflow

# Ensure the correct experiment is set
mlflow.set_tracking_uri("http://34.28.175.237:5000")
mlflow.set_experiment("MLflow Connectivity Test")

# Check if there's already an active run
if mlflow.active_run():
    mlflow.end_run()  # Ends any active run before starting a new one

with mlflow.start_run():
    mlflow.log_param("test_param", 1)
    mlflow.log_metric("test_metric", 0.99)
    print("MLflow run started successfully!")
