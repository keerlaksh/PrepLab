import pandas as pd


def check_symmetry(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.5
) -> dict:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if threshold < 0:
        raise ValueError("Threshold cannot be negative.")

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    results = {}

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist.")

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numerical.")

        skewness = df[col].dropna().skew()

        if pd.isna(skewness):
            classification = "undetermined"
        elif abs(skewness) <= threshold:
            classification = "approximately_symmetric"
        else:
            classification = "not_symmetric"

        results[col] = {
            "skewness": None if pd.isna(skewness) else float(skewness),
            "is_symmetric": classification == "approximately_symmetric",
            "classification": classification
        }

    return results