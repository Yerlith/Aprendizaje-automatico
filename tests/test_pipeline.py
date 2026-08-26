from src.data import load_data
from src.models import train_models


def test_dataset_is_not_empty():
    features, target = load_data()

    assert not features.empty
    assert not target.empty
    assert len(features) == len(target)


def test_models_return_valid_predictions():
    features, target = load_data()
    results = train_models()

    assert set(results) == {"random_forest", "logistic_regression"}
    for result in results.values():
        predictions = result["model"].predict(features.head(10))
        assert len(predictions) == 10
        assert set(predictions).issubset(set(target.unique()))
        assert 0 <= result["accuracy"] <= 1