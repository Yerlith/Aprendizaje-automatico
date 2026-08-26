from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

from src.data import load_data


def calculate_metrics(y_true, y_pred):
    """Calculate weighted F1-Score and precision for model predictions."""
    return {
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
    }


def train_models(
    dataset_path="data/mobile_prices.csv",
    test_size=0.2,
    random_state=42,
):
    """Train and evaluate Random Forest and Logistic Regression classifiers."""
    features, target = load_data(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
    }
    results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, predictions),
            **calculate_metrics(y_test, predictions),
        }

    return results