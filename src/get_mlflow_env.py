import mlflow
import os
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def get_run_env_file(experiment_name="Apple_Models", 
                     run_name="first_run", 
                     output_dir="./src", 
                     port=8080):
    """
    Copy python_env.yaml, conda.yaml and requirements.txt directly from MLflow artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"http://127.0.0.1:{port}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise Exception(f"Experiment {experiment_name} not found")

    logging.info(f"Found experiment: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    print(f"{run_name = }")
    if len(runs) == 0:
        raise Exception(f"Run {run_name} not found in experiment {experiment_name}")

    run_id = runs.iloc[0].run_id
    logging.info(f"Found run: {run_name} (ID: {run_id})")

    # Get artifact URI
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    if artifact_uri.startswith("file://"):
        artifact_uri = artifact_uri[7:]

    # Get the rf_apples directory path
    rf_apples_path = os.path.join(artifact_uri, "rf_apples")

    # Copy python_env.yaml
    python_env_src = os.path.join(rf_apples_path, "python_env.yaml")
    if os.path.exists(python_env_src):
        shutil.copy2(python_env_src, output_path / "python_env.yaml")
        logging.info(f"Copied python_env.yaml to {output_dir}")

    # Copy conda.yaml if it exists
    conda_src = os.path.join(rf_apples_path, "conda.yaml")
    if os.path.exists(conda_src):
        shutil.copy2(conda_src, output_path / "conda.yaml")
        logging.info(f"Copied conda.yaml to {output_dir}")

    # Copy requirements.txt
    requirements_src = os.path.join(rf_apples_path, "requirements.txt")
    if not os.path.exists(requirements_src):
        raise Exception("requirements.txt not found in artifacts")
    shutil.copy2(requirements_src, output_path / "requirements.txt")
    logging.info(f"Copied requirements.txt to {output_dir}")

    # Check if at least one environment file was found
    if not os.path.exists(python_env_src) and not os.path.exists(conda_src):
        raise Exception("Neither python_env.yaml nor conda.yaml found in artifacts")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve MLflow run environment files")
    parser.add_argument("--experiment", default="Apple_Models", help="Name of the MLflow experiment")
    parser.add_argument("--run", default="first_run", help="Name of the run")
    parser.add_argument("--output", default="./src", help="Output directory for the files")
    parser.add_argument("--port", type=int, default=8080, help="MLflow server port")

    args = parser.parse_args()

    get_run_env_file(
        experiment_name=args.experiment,
        run_name=args.run,
        output_dir=args.output,
        port=args.port
    )