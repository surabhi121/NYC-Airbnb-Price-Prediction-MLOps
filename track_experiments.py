"""Experiment tracking (implementation.md, Task 7): train the five candidate configs and log each as an MLflow run.

Metrics are on the dollar scale (features.evaluate), and every logged model
carries its own log1p/expm1 target transform.
"""
import mlflow
import mlflow.sklearn
from mlflow.models import Model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from features import RANDOM_STATE, build_model, clean_data, evaluate, load_data, split_data
from registry import EXPERIMENT_NAME, require_tracking_uri

# run name -> (model class, params). Params are logged to MLflow verbatim.
CONFIGS = {
    "linreg_baseline": (LinearRegression, {}),
    "rf_100": (RandomForestRegressor, {"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_STATE}),
    "rf_300_depth10": (RandomForestRegressor, {"n_estimators": 300, "max_depth": 10, "n_jobs": -1, "random_state": RANDOM_STATE}),
    "gb_100_lr01": (GradientBoostingRegressor, {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE}),
    "gb_200_lr005": (GradientBoostingRegressor, {"n_estimators": 200, "learning_rate": 0.05, "random_state": RANDOM_STATE}),
}


# MLflow 3 saves sklearn models with skops, which only loads allow-listed
# types. The tree ensembles store their trees in sklearn's Tree object, which
# skops blocks unless trusted. We trust exactly that one type, since we create
# these files ourselves; the list is saved in the MLmodel file and reused by
# mlflow.sklearn.load_model, so the API needs no extra setting.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def train_and_log(run_name, X_train, X_test, y_train, y_test):
    """Fit one config inside an MLflow run. Returns (run_id, metrics)."""
    model_class, params = CONFIGS[run_name]
    with mlflow.start_run(run_name=run_name) as run:
        model = build_model(model_class(**params)).fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_params(
            {"model_type": model_class.__name__, "target_transform": "log1p/expm1", **params}
        )
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        # The API downloads and holds the champion in memory, so size is a
        # selection criterion (registry.MAX_MODEL_SIZE_MB). MLflow records the
        # size in the MLmodel file; reading it doesn't download the model.
        size_bytes = Model.load(model_info.model_uri).model_size_bytes
        metrics["model_size_mb"] = round(size_bytes / 1e6, 1)
        mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
    return run.info.run_id, metrics


def main():
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    splits = split_data(clean_data(load_data()))
    for run_name in CONFIGS:
        run_id, m = train_and_log(run_name, *splits)
        print(
            f"{run_name:16s} rmse={m['rmse']:7.2f}  mae={m['mae']:6.2f}  r2={m['r2']:.3f}  "
            f"size={m['model_size_mb']:6.1f}MB  run_id={run_id}"
        )


if __name__ == "__main__":
    main()
