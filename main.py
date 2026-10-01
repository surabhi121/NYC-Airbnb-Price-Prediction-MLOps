"""FastAPI service for AirbnbPriceModel@champion.

The model is loaded once at startup from the MLflow registry. mlflow reads
MLFLOW_TRACKING_URI from the environment, so the same image works against a
laptop server, CI, or the Compose `mlflow-server` service.
"""
import logging
import os
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from schemas import Listing, PricePrediction

MODEL_URI = os.environ.get("MODEL_URI", "models:/AirbnbPriceModel@champion")
logger = logging.getLogger("uvicorn.error")


def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # If the MLflow server is unreachable, mlflow retries with backoff before
    # failing (~4 min with its defaults; the Dockerfile shortens this), so say
    # what we're waiting on instead of hanging silently.
    logger.info("Loading %s from %s", MODEL_URI, mlflow.get_tracking_uri())
    app.state.model = load_model()
    logger.info("Model loaded")
    yield


app = FastAPI(title="NYC Airbnb Price API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PricePrediction)
def predict(listing: Listing) -> PricePrediction:
    features = pd.DataFrame([listing.model_dump()])
    # The champion is a TransformedTargetRegressor: .predict() already applies
    # expm1 to undo the log1p it was trained on, so this value is in USD.
    price = float(app.state.model.predict(features)[0])
    return PricePrediction(predicted_price=round(max(price, 0.0), 2))
