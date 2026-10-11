import pandas as pd


def type_correction(
    df: pd.DataFrame,
    conversions: dict[str, str],
) -> dict:
    """
    Convert specified columns to explicitly requested data types.

    Supported types:
        numeric, integer, float, string, category, datetime, boolean

    Invalid values for numeric, integer, datetime, or boolean conversions
    are reported and converted to missing values where applicable.
    """
    if df is None:
        return {"success": False, "message": "DataFrame is None."}

    if not conversions:
        return {
            "success": False,
            "message": "Provide at least one column and target type.",
        }

    result_df = df.copy()
    conversion_report = []

    aliases = {
        "number": "numeric",
        "int": "integer",
        "str": "string",
        "date": "datetime",
        "bool": "boolean",
    }

    allowed_types = {
        "numeric", "integer", "float",
        "string", "category", "datetime", "boolean",
    }

    try:
        for column, requested_type in conversions.items():
            if column not in result_df.columns:
                return {
                    "success": False,
                    "message": f"Column not found: {column}",
                }

            target_type = aliases.get(
                requested_type.strip().lower(),
                requested_type.strip().lower(),
            )

            if target_type not in allowed_types:
                return {
                    "success": False,
                    "message": (
                        f"Unsupported type '{requested_type}' for {column}. "
                        f"Supported types: {sorted(allowed_types)}"
                    ),
                }

            original = result_df[column].copy()
            originally_missing = original.isna()
            failed_count = 0

            if target_type in {"numeric", "float", "integer"}:
                converted = pd.to_numeric(original, errors="coerce")

                if target_type == "integer":
                    fractional = converted.notna() & (
                        (converted % 1) != 0
                    )
                    failed_count = int(
                        (converted.isna() & ~originally_missing).sum()
                        + fractional.sum()
                    )
                    converted = converted.mask(fractional).astype("Int64")

                elif target_type == "float":
                    converted = converted.astype("float64")
                    failed_count = int(
                        (converted.isna() & ~originally_missing).sum()
                    )

                else:
                    failed_count = int(
                        (converted.isna() & ~originally_missing).sum()
                    )

                result_df[column] = converted

            elif target_type == "datetime":
                converted = pd.to_datetime(original, errors="coerce")
                failed_count = int(
                    (converted.isna() & ~originally_missing).sum()
                )
                result_df[column] = converted

            elif target_type == "boolean":
                boolean_map = {
                    "true": True, "1": True, "yes": True, "y": True,
                    "false": False, "0": False, "no": False, "n": False,
                }

                def convert_boolean(value):
                    if pd.isna(value):
                        return pd.NA
                    if isinstance(value, bool):
                        return value
                    return boolean_map.get(str(value).strip().lower(), pd.NA)

                converted = original.map(convert_boolean).astype("boolean")
                failed_count = int(
                    (converted.isna() & ~originally_missing).sum()
                )
                result_df[column] = converted

            elif target_type == "string":
                result_df[column] = original.astype("string")

            elif target_type == "category":
                result_df[column] = original.astype("category")

            conversion_report.append({
                "column": column,
                "original_dtype": str(original.dtype),
                "requested_type": target_type,
                "result_dtype": str(result_df[column].dtype),
                "failed_conversions": failed_count,
            })

        return {
            "success": True,
            "dataframe": result_df,
            "conversion_report": conversion_report,
            "original_shape": df.shape,
            "final_shape": result_df.shape,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Data-type correction failed: {e}",
        }