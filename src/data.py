from pathlib import Path

import pandas as pd


def load_data(dataset_path="data/mobile_prices.csv"):
    """Load the mobile prices dataset and split features from the target."""
    data = pd.read_csv(Path(dataset_path))
    if "price_range" not in data.columns:
        raise ValueError("The dataset must contain a 'price_range' column")

    return data.drop(columns="price_range"), data["price_range"]