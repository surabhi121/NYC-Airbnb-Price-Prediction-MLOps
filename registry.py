"""MLflow naming and Model Registry helpers.

Uses registered-model aliases (@champion), not the deprecated stages API.
"""
import math
import os
import sys

import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "airbnb-price-prediction"
MODEL_NAME = "AirbnbPriceModel"
CHAMPION_ALIAS = "champion"

# Selection rule: the lowest RMSE on the held-out test set wins, among models
# no bigger than MAX_MODEL_SIZE_MB.
# - RMSE decides because it punishes large dollar misses hardest: for a
#   pricing tool, one $300 miss hurts more than three $100 misses. MAE is
#   checked too, so a disagreement is printed rather than silently ignored.
# - The size budget exists because the API downloads the champion at startup
#   and keeps it in memory. In the first experiment round rf_100 had the best
#   RMSE ($77.53) but was 326 MB; rf_300_depth10 was $1.22 worse at 39 MB.
SELECTION_METRIC = "rmse"
MAX_MODEL_SIZE_MB = 100


def require_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        sys.exit(
            "MLFLOW_TRACKING_URI is not set. Start the server (see implementation.md, Task 7) and run:\n"
            "  export MLFLOW_TRACKING_URI=http://127.0.0.1:5001"
        )
    return uri


def pick_best(results: dict[str, dict[str, float]]) -> str:
    """results maps run_id -> metrics (incl. model_size_mb). Returns the winning run_id."""
    eligible = {
        run_id: m for run_id, m in results.items() if m["model_size_mb"] <= MAX_MODEL_SIZE_MB
    }
    if not eligible:
        raise ValueError(f"No model within the {MAX_MODEL_SIZE_MB} MB size budget")

    best = min(eligible, key=lambda run_id: eligible[run_id][SELECTION_METRIC])
    best_overall = min(results, key=lambda run_id: results[run_id][SELECTION_METRIC])
    if best_overall != best:
        print(f"note: {best_overall} has a lower RMSE but is over the {MAX_MODEL_SIZE_MB} MB budget")
    best_by_mae = min(eligible, key=lambda run_id: eligible[run_id]["mae"])
    if best_by_mae != best:
        print(f"note: lowest-RMSE run {best} differs from lowest-MAE run {best_by_mae}; RMSE decides")
    return best


def best_run_id(experiment_name: str) -> str:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
    )
    results = {
        row["run_id"]: {
            "rmse": row["metrics.rmse"],
            "mae": row["metrics.mae"],
            "r2": row["metrics.r2"],
            # Runs logged before the size metric existed can't prove they fit the budget.
            "model_size_mb": math.inf if math.isnan(row["metrics.model_size_mb"]) else row["metrics.model_size_mb"],
        }
        for _, row in runs.iterrows()
    }
    return pick_best(results)


def logged_model_uri(run_id: str) -> str:
    """MLflow 3 stores a run's model as its own LoggedModel (models:/m-...), not as a run artifact."""
    outputs = MlflowClient().get_run(run_id).outputs.model_outputs
    if not outputs:
        raise ValueError(f"Run {run_id} has no logged model")
    return f"models:/{outputs[0].model_id}"


def register_and_promote(run_id: str) -> str:
    """Register the run's model and point @champion at it. Returns the new version."""
    version = mlflow.register_model(logged_model_uri(run_id), MODEL_NAME).version
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    return version


if __name__ == "__main__":
    require_tracking_uri()
    run_id = best_run_id(EXPERIMENT_NAME)
    version = register_and_promote(run_id)
    print(f"registered {MODEL_NAME} v{version} from run {run_id} -> @{CHAMPION_ALIAS}")
