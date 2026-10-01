"""Prefect flow (implementation.md, Task 10): load -> split -> train_and_log x5 -> promote best -> request deploy.

MLFLOW_TRACKING_URI is read from the environment. Run once:
    python orchestrate_training.py
Serve on a weekly schedule (needs `prefect server start` running):
    python orchestrate_training.py --serve
"""
import os
import sys

import mlflow
from prefect import flow, get_run_logger, task

import features
import track_experiments
from registry import EXPERIMENT_NAME, pick_best, register_and_promote, require_tracking_uri
from scripts.trigger_deploy import trigger_deploy


@task(retries=2, retry_delay_seconds=5)
def load_data():
    return features.clean_data(features.load_data())


@task
def split_data(df):
    return features.split_data(df)


@task
def train_and_log(run_name, splits):
    run_id, metrics = track_experiments.train_and_log(run_name, *splits)
    get_run_logger().info(
        "%s rmse=%.2f mae=%.2f r2=%.3f size=%.1fMB",
        run_name, metrics["rmse"], metrics["mae"], metrics["r2"], metrics["model_size_mb"],
    )
    return run_id, metrics


@task
def promote_best_model(results):
    run_id = pick_best(results)
    version = register_and_promote(run_id)
    get_run_logger().info("promoted run %s as AirbnbPriceModel v%s @champion", run_id, version)
    return version


@task
def request_deploy(version):
    repo, token = os.environ.get("GITHUB_REPO"), os.environ.get("GITHUB_TOKEN")
    if not (repo and token):
        get_run_logger().warning("GITHUB_REPO/GITHUB_TOKEN not set; skipping deploy trigger")
        return False
    return trigger_deploy(version, repo, token)


@flow(name="airbnb-price-training")
def training_flow():
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    splits = split_data(load_data())
    results = {}
    for run_name in track_experiments.CONFIGS:
        run_id, metrics = train_and_log(run_name, splits)
        results[run_id] = metrics
    version = promote_best_model(results)
    request_deploy(version)
    return version


if __name__ == "__main__":
    if "--serve" in sys.argv:
        training_flow.serve(name="weekly-retrain", cron="0 3 * * 1")  # Mondays 03:00
    else:
        training_flow()
