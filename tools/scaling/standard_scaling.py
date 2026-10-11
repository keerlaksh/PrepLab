import pandas as pd
from sklearn.preprocessing import StandardScaler


def standard_scaling(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include="number").columns.tolist()

    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' does not exist.")
        if not pd.api.types.is_numeric_dtype(result[col]):
            raise TypeError(f"Column '{col}' is not numerical.")
        if result[col].isna().any():
            raise ValueError(f"Impute missing values in '{col}' first.")

    if columns:
        scaler = StandardScaler()
        result[columns] = scaler.fit_transform(result[columns])

    return result
