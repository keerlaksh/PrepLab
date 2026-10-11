import pandas as pd


def iqr_strategy(series: pd.Series) -> dict:
    values = series.dropna()

    if values.empty:
        return {
            "outlier_indices": [],
            "outlier_count": 0,
            "lower_bound": None,
            "upper_bound": None
        }

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (series < lower_bound) | (series > upper_bound)

    return {
        "outlier_indices": series.index[mask.fillna(False)].tolist(),
        "outlier_count": int(mask.sum()),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound)
    }