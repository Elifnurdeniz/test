import mlflow

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://34.28.175.237:5000")
experiment_name = "MLflow Connectivity Test"

# Get experiment details
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    print(f"Experiment '{experiment_name}' found with ID: {experiment.experiment_id}")

    # Fetch all runs associated with the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if not runs.empty:
        print("Existing Runs:")
        print(runs[["run_id", "status", "start_time"]])  # Print only relevant columns
    else:
        print("No previous runs found.")
else:
    print(f"Experiment '{experiment_name}' not found.")

# Start a new run
if mlflow.active_run():
    mlflow.end_run()

mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    print(f"New Run ID: {run.info.run_id}")
    mlflow.log_param("test_param", 1)
    mlflow.log_metric("test_metric", 0.99)
    print("MLflow run started successfully!")
