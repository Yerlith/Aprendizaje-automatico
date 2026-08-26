from pathlib import Path
import sys

import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import train_models


def main():
    """Train both classifiers and log the best one to MLflow."""
    mlflow.set_experiment("mobile-price-classification")
    results = train_models()
    best_name = max(results, key=lambda name: results[name]["accuracy"])
    best_result = results[best_name]

    with mlflow.start_run(run_name="model-comparison"):
        for name, result in results.items():
            mlflow.log_metric(f"{name}_accuracy", result["accuracy"])

        mlflow.log_param("best_model", best_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("best_accuracy", best_result["accuracy"])
        mlflow.sklearn.log_model(best_result["model"], "best_model")

    print(f"Best model: {best_name}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")


if __name__ == "__main__":
    main()