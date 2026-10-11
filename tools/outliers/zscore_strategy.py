import pandas as pd


def zscore_strategy(
    series: pd.Series,
    threshold: float = 3.0
) -> dict:
    values = series.dropna()

    if values.empty:
        return {
            "outlier_indices": [],
            "outlier_count": 0,
            "lower_bound": None,
            "upper_bound": None
        }

    mean_value = values.mean()
    std_value = values.std(ddof=0)

    if std_value == 0:
        return {
            "outlier_indices": [],
            "outlier_count": 0,
            "lower_bound": float(mean_value),
            "upper_bound": float(mean_value)
        }

    z_scores = (series - mean_value) / std_value
    mask = z_scores.abs() > threshold

    return {
        "outlier_indices": series.index[mask.fillna(False)].tolist(),
        "outlier_count": int(mask.sum()),
        "lower_bound": float(mean_value - threshold * std_value),
        "upper_bound": float(mean_value + threshold * std_value)
    }