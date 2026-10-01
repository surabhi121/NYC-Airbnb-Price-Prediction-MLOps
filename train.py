"""Baseline (implementation.md, Task 4): LinearRegression trained on log1p(price), scored in dollars.

features.build_model wraps the pipeline in a TransformedTargetRegressor that
fits on log1p(price) and applies expm1 on predict, so the metrics printed
here are real USD errors. If RMSE ever comes out in the thousands (or below
1), the log/exp inversion is the first thing to check.
"""
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from features import build_model, clean_data, evaluate, load_data, split_data

MODEL_PATH = Path("models/model.pkl")


def main():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"rows after cleaning: {len(df)}  (train={len(X_train)}, test={len(X_test)})")

    model = build_model(LinearRegression()).fit(X_train, y_train)
    for name, value in evaluate(model, X_test, y_test).items():
        print(f"{name}: {value:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    assert (reloaded.predict(X_test.head()) == model.predict(X_test.head())).all()
    print(f"saved and reloaded {MODEL_PATH}")


if __name__ == "__main__":
    main()
