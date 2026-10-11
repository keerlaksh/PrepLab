import pandas as pd


def frequency_encoding(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    result = df.copy()

    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' does not exist.")

        frequencies = result[col].value_counts(normalize=True)
        result[col] = result[col].map(frequencies)

    return result