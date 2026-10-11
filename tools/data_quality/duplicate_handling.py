import pandas as pd


def duplicate_handling(
    df: pd.DataFrame,
    remove: bool = False,
    subset: list[str] | None = None,
    keep: str | bool = "first",
) -> dict:
    """
    Detect duplicate rows and optionally remove them.

    Args:
        df: Input DataFrame.
        remove: If True, remove duplicate rows.
        subset: Columns to compare. None means compare all columns.
        keep: Which duplicate occurrence to retain:
              "first", "last", or False.

    Returns:
        A dictionary containing the resulting DataFrame and an audit report.
    """
    if df is None:
        return {"success": False, "message": "DataFrame is None."}

    if keep not in ("first", "last", False):
        return {
            "success": False,
            "message": "keep must be 'first', 'last', or False.",
        }

    if subset is not None:
        missing_columns = [col for col in subset if col not in df.columns]
        if missing_columns:
            return {
                "success": False,
                "message": f"Columns not found: {missing_columns}",
            }

    try:
        duplicate_mask = df.duplicated(subset=subset, keep=keep)
        duplicate_count = int(duplicate_mask.sum())

        result_df = (
            df.drop_duplicates(subset=subset, keep=keep).copy()
            if remove
            else df.copy()
        )

        return {
            "success": True,
            "dataframe": result_df,
            "remove_requested": remove,
            "duplicate_count": duplicate_count,
            "rows_removed": len(df) - len(result_df),
            "original_shape": df.shape,
            "final_shape": result_df.shape,
            "subset": subset,
            "keep": keep,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Duplicate handling failed: {e}",
        }