import pandas as pd


def mode_imputation(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    result = df.copy()

    if columns is None:
        columns = result.columns[result.isna().any()].tolist()

    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' does not exist.")

        modes = result[col].mode(dropna=True)
        if not modes.empty:
            result[col] = result[col].fillna(modes.iloc[0])

    return result