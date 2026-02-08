from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

import mlflow
import pandas as pd

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

if TYPE_CHECKING:
    from mlflow.tracing import MlflowClient


@dataclass
class MLFlowParam:
    """Parameters for the MLFlow"""
    experiment_name: str = "RandomizedSearchCV_Random_Forest"
    uri: str = "http://127.0.0.1:8080"


@dataclass(frozen=True)
class DataConfig:
    """Data path and parameters"""
    path: str = "data/fake_data.csv"


@dataclass
class ModelConfig:
    """Parameters for the RandomizedSearchCV"""
    param_distributions: dict[str, Any] = field(
        default_factory=lambda: {
            'n_estimators': randint(50, 200),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 4),
            })
    n_iter: int = 5
    cv: int = 5
    scoring: str = 'r2'
    random_state: int = 42


@dataclass
class Configs:
    """Seting up confihuration"""
    ml_flow = MLFlowParam()
    data = DataConfig()
    model_params = ModelConfig()


def resolve_experiment_name(client: "MlflowClient", base_name: str) -> str:
    """Get the base name for the experiment"""
    experiment = client.get_experiment_by_name(base_name)

    if experiment and experiment.lifecycle_stage == "deleted":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    return base_name


def load_and_prep_data(data_path: str) -> list[Any]:
    """Load and prepare data for training."""
    data = pd.read_csv(data_path)
    x = data.drop(columns=["date", "demand"])
    x = x.astype('float')
    y = data["demand"]
    return train_test_split(x, y, test_size=0.2, random_state=42)


def get_best_parent_run(runs: Any) -> Any:
    """
    :runs: mlflow.store.entities.paged_list.PagedList
    :return: mlflow.entities.run.Run
    """
    parent_run = None
    for run in runs:
        if 'best_n_estimators' in run.data.params:
            parent_run = run
            break

    return parent_run


def find_best_run_from_parent(runs: Any, parent_run: Any) -> Any:
    """
    Looking into runs to get the best one  
    :runs: mlflow.store.entities.paged_list.PagedList  
    :parent_run: mlflow.entities.run.Run  

    :return: mlflow.entities.run.Run
    """
    best_run = None
    if parent_run:
        # Extract best parameters from parent run
        best_params_from_parent = {
            'n_estimators': parent_run.data.params[
                'best_n_estimators'],
            'max_depth': parent_run.data.params[
                'best_max_depth'],
            'min_samples_split': parent_run.data.params[
                'best_min_samples_split'],
            'min_samples_leaf': parent_run.data.params[
                'best_min_samples_leaf']
        }

        # Find the child run with these parameters
        for run in runs:
            if ('n_estimators' in run.data.params and
                run.data.params['n_estimators'] ==
                    best_params_from_parent['n_estimators'] and
                run.data.params['max_depth'] ==
                    best_params_from_parent['max_depth'] and
                run.data.params['min_samples_split'] ==
                    best_params_from_parent['min_samples_split'] and
                run.data.params['min_samples_leaf'] ==
                    best_params_from_parent['min_samples_leaf']):
                best_run = run
                break
    return best_run


def save_trial_summary(experiment_name: str, search: Any, parent_run: Any,
                       best_run_name: str) -> None:
    """
    :experiment_name: <class 'str'>  
    :search: <class 'sklearn.model_selection._search.RandomizedSearchCV'>  
    :parent_run: <class 'mlflow.entities.run.Run'>  
    :best_run_name: <class 'str'>  
    """
    summary = f"""Random Forest Trials Summary:
    ---------------------------
    Best Experiment Name: {experiment_name}
    Best Run Name: {best_run_name}
    Best Model Parameters:
    Number of Trees: {search.best_params_['n_estimators']}
    Max Tree Depth: {search.best_params_['max_depth']}
    Min Samples Split: {search.best_params_['min_samples_split']}
    Min Samples Leaf: {search.best_params_['min_samples_leaf']}
    Best CV Score: {search.best_score_:.4f}"""

    # Log summary to the parent run
    with mlflow.start_run(run_id=parent_run.info.run_id):
        # Log summary as an artifact
        with open("summary.txt", "w", encoding="utf8") as f:
            f.write(summary)
        mlflow.log_artifact("summary.txt")


def perform_random_search_cv(cfg: Configs, x_train: pd.DataFrame,
                             y_train: pd.Series) -> Any:
    """
    return: <class 'sklearn.model_selection._search.RandomizedSearchCV'>
    """
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=cfg.model_params.random_state),
        param_distributions=cfg.model_params.param_distributions,
        n_iter=cfg.model_params.n_iter,
        cv=cfg.model_params.cv,
        scoring=cfg.model_params.scoring,
        random_state=cfg.model_params.random_state
    )

    # Fit the model - autolog will automatically create the runs
    search.fit(x_train, y_train)

    return search


def main():
    """Get the best of the best!"""
    # Basic setup
    cfg = Configs()

    # Set up MLflow tracking
    mlflow.set_tracking_uri(cfg.ml_flow.uri)

    # Handle experiment creation/deletion
    client = mlflow.tracking.MlflowClient()

    experiment_name = resolve_experiment_name(
        client, cfg.ml_flow.experiment_name)

    mlflow.set_experiment(experiment_name)

    # Enable autologging
    mlflow.sklearn.autolog(log_models=True)

    # Load data
    X_train, _, y_train, _ = load_and_prep_data(cfg.data.path)

    # Create and run RandomizedSearchCV
    search = perform_random_search_cv(cfg, X_train, y_train)

    # Find the best run from MLflow
    runs = client.search_runs(
        experiment_ids=[
            client.get_experiment_by_name(experiment_name).experiment_id],
        filter_string="",
        max_results=50
    )

    # Identify the parent and its best run parameters
    parent_run = get_best_parent_run(runs)

    best_run = find_best_run_from_parent(runs, parent_run)

    best_run_name = best_run.data.tags.get(
        'mlflow.runName', 'Not found') if best_run else 'Not found'

    # Create a summary of results with better formatting
    save_trial_summary(experiment_name, search, parent_run, best_run_name)


if __name__ == "__main__":
    main()
