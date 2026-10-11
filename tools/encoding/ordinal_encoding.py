import pandas as pd


def ordinal_encoding(
    df: pd.DataFrame,
    mappings: dict[str, dict]
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    result = df.copy()

    for col, mapping in mappings.items():
        if col not in result.columns:
            raise ValueError(f"Column '{col}' does not exist.")

        unknown = set(result[col].dropna().unique()) - set(mapping.keys())
        if unknown:
            raise ValueError(f"Unmapped categories in '{col}': {unknown}")

        result[col] = result[col].map(mapping)

    return result