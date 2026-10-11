import pandas as pd


def outlier_capping(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if factor < 0:
        raise ValueError("factor cannot be negative.")

    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include="number").columns.tolist()

    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' does not exist.")

        if not pd.api.types.is_numeric_dtype(result[col]):
            raise TypeError(f"Column '{col}' is not numerical.")

        values = result[col].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        result[col] = result[col].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return result