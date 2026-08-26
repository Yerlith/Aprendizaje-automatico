from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def train_random_forest(
    dataset_path="data/mobile_prices.csv",
    experiment_name="mobile-price-classification",
    n_estimators=100,
    test_size=0.2,
    random_state=42,
):
    """Train a Random Forest and log its run, metrics, and model in MLflow."""
    data = pd.read_csv(Path(dataset_path))
    if "price_range" not in data.columns:
        raise ValueError("The dataset must contain a 'price_range' column")

    features = data.drop(columns="price_range")
    target = data["price_range"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision_weighted": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
        }

        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "test_size": test_size,
                "random_state": random_state,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_model")

    return model, metrics