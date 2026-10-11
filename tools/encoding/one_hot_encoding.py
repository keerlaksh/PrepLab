import pandas as pd


def one_hot_encoding(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    result = df.copy()

    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' does not exist.")

    return pd.get_dummies(
        result,
        columns=columns,
        dtype=int
    )