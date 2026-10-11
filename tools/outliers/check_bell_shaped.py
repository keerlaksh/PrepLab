import pandas as pd
import numpy as np
from scipy.stats import shapiro, skew
from typing import Any


def check_bell_shaped(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Check whether numerical columns are approximately bell-shaped.

    Uses:
    - Shapiro-Wilk test for normality
    - Skewness to measure distribution asymmetry

    Returns a dictionary containing the results for each column.
    """

    results = {}

    if columns is None:
        columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        values = values[np.isfinite(values)]

        if len(values) < 3:
            results[column] = {
                "is_bell_shaped": False,
                "reason": "Insufficient observations",
                "sample_size": len(values),
                "skewness": None,
                "p_value": None,
            }
            continue

        if values.nunique() < 2:
            results[column] = {
                "is_bell_shaped": False,
                "reason": "Constant or identical values",
                "sample_size": len(values),
                "skewness": 0.0,
                "p_value": 0.0,
            }
            continue

        column_skewness = float(skew(values))

        # Shapiro-Wilk accepts at most 5000 observations
        # for this implementation.
        test_values = (
            values.sample(n=5000, random_state=42)
            if len(values) > 5000
            else values
        )

        try:
            _, p_value = shapiro(test_values)
            p_value = float(p_value)

            # A practical approximation, not proof of normality.
            is_bell_shaped = (
                p_value >= alpha
                and abs(column_skewness) <= 0.5
            )

            results[column] = {
                "is_bell_shaped": bool(is_bell_shaped),
                "reason": (
                    "Approximately bell-shaped"
                    if is_bell_shaped
                    else "Distribution may be non-normal or skewed"
                ),
                "sample_size": len(values),
                "skewness": column_skewness,
                "p_value": p_value,
            }

        except Exception as exc:
            results[column] = {
                "is_bell_shaped": False,
                "reason": f"Normality test failed: {exc}",
                "sample_size": len(values),
                "skewness": column_skewness,
                "p_value": None,
            }

    return {
        "results": results,
        "alpha": alpha,
        "method": "Shapiro-Wilk test and skewness",
    }