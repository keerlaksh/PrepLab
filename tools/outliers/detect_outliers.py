import pandas as pd

from outliers.iqr_strategy import iqr_strategy
from outliers.zscore_strategy import zscore_strategy


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr"
) -> dict:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    if method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'.")

    results = {}

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist.")

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numerical.")

        if method == "iqr":
            results[col] = iqr_strategy(df[col])
        else:
            results[col] = zscore_strategy(df[col])

    return {
        "method": method,
        "results": results
    }