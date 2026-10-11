
# tools/inspect_data.py

import pandas as pd


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Inspect a dataset without modifying it.

    Checks:
    1. Duplicate-row count
    2. Categorical columns (object dtype)
    3. Numerical columns (dtype other than object)
    4. Categorical cardinality (unique values per column)
    """

    if df is None or df.empty:
        return {
            "success": False,
            "message": "Dataset is empty."
        }

    try:
        # 1. Detect duplicate rows
        duplicate_count = int(df.duplicated().sum())

        # 2. Identify categorical columns
        categorical_columns = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # 3. Identify columns with dtype other than object
        numerical_columns = [
            col for col in df.columns
            if df[col].dtype != "object"
        ]

        # 4. Calculate categorical cardinality
        cardinality = {
            col: int(df[col].nunique())
            for col in categorical_columns
        }

        return {
            "success": True,
            "duplicate_count": duplicate_count,
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            "categorical_cardinality": cardinality
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Data inspection failed: {str(e)}"
        }