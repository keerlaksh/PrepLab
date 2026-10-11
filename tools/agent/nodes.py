# agent/nodes.py

from typing import Any

import pandas as pd

from agent.state import DataState

# Initial preprocessing tools
from problem_definition import problem_definition
from inspect_data import inspect_data
from clean_columns import clean_columns
from data_quality.duplicate_handling import duplicate_handling
from data_quality.type_correction import type_correction

# Imputation tools
from imputation.mean_imputation import mean_imputation
from imputation.median_imputation import median_imputation
from imputation.mode_imputation import mode_imputation
from imputation.check_skewness import check_skewness
from imputation.check_symmetry import check_symmetry

# Outlier tools
from outliers.detect_outliers import detect_outliers
from outliers.outlier_capping import outlier_capping
from outliers.check_bell_shaped import check_bell_shaped

# Encoding tools
from encoding.one_hot_encoding import one_hot_encoding
from encoding.ordinal_encoding import ordinal_encoding
from encoding.frequency_encoding import frequency_encoding

# Scaling tools
from scaling.standard_scaling import standard_scaling
from scaling.minmax_scaling import minmax_scaling
from scaling.robust_scaling import robust_scaling


# ==================================================
# 1. HISTORY HELPER
# ==================================================

def update_history(
    state: DataState,
    step: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:

    entry = {
        "step": step,
        "status": status,
        "details": details or {},
    }

    history = state.get("processing_history", []).copy()
    history.append(entry)

    updates: dict[str, Any] = {
        "processing_history": history,
    }

    if status == "success":
        completed = state.get("completed_steps", []).copy()

        if step not in completed:
            completed.append(step)

        updates["completed_steps"] = completed

    return updates


def record_failure(
    state: DataState,
    step: str,
    error: Exception,
) -> dict[str, Any]:

    return update_history(
        state,
        step,
        "failed",
        {"message": str(error)},
    )


# ==================================================
# 2. PROBLEM DEFINITION
# ==================================================

def problem_definition_node(state: DataState) -> dict[str, Any]:
    try:
        result = problem_definition(state["current_df"])

        if not result.get("success", False):
            return update_history(
                state,
                "problem_definition",
                "failed",
                {
                    "message": result.get(
                        "message",
                        "Problem definition failed.",
                    )
                },
            )

        return {
            "problem_summary": result.get("problem_summary", ""),
            **update_history(
                state,
                "problem_definition",
                "success",
                {
                    "problem_summary": result.get(
                        "problem_summary", ""
                    ),
                    "metadata": result.get("metadata", {}),
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "problem_definition", e)


# ==================================================
# 3. INSPECT DATA
# ==================================================

def inspect_data_node(state: DataState) -> dict[str, Any]:
    try:
        result = inspect_data(state["current_df"])

        if not result.get("success", False):
            return {
                "inspection_results": result,
                **update_history(
                    state,
                    "inspect_data",
                    "failed",
                    {
                        "message": result.get(
                            "message",
                            "Inspection failed.",
                        )
                    },
                ),
            }

        return {
            "inspection_results": result,
            **update_history(
                state,
                "inspect_data",
                "success",
                {
                    "duplicate_count": result.get(
                        "duplicate_count", 0
                    ),
                    "categorical_columns": result.get(
                        "categorical_columns", []
                    ),
                    "numerical_columns": result.get(
                        "numerical_columns", []
                    ),
                    "categorical_cardinality": result.get(
                        "categorical_cardinality", {}
                    ),
                },
            ),
        }

    except Exception as e:
        return {
            "inspection_results": {
                "success": False,
                "message": str(e),
            },
            **record_failure(state, "inspect_data", e),
        }


# ==================================================
# 4. CLEAN COLUMNS
# ==================================================

def clean_columns_node(state: DataState) -> dict[str, Any]:
    try:
        result = clean_columns(
            df=state["current_df"],
            target_column=state.get("target_column"),
            problem_summary=state.get("problem_summary", ""),
        )

        if not result.get("success", False):
            return update_history(
                state,
                "clean_columns",
                "failed",
                {
                    "message": result.get(
                        "message",
                        "Column cleaning failed.",
                    )
                },
            )

        return {
            "current_df": result["cleaned_df"],
            **update_history(
                state,
                "clean_columns",
                "success",
                {
                    "original_shape": result.get("original_shape"),
                    "final_shape": result.get("final_shape"),
                    "dropped_columns": result.get(
                        "dropped_columns", []
                    ),
                    "review_flags": result.get(
                        "review_flags", {}
                    ),
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "clean_columns", e)


# ==================================================
# 5. DUPLICATE HANDLING
# ==================================================

def duplicate_handling_node(state: DataState) -> dict[str, Any]:
    try:
        result = duplicate_handling(
            df=state["current_df"],
            remove=True,
        )

        if not result.get("success", False):
            return update_history(
                state,
                "duplicate_handling",
                "failed",
                {
                    "message": result.get(
                        "message",
                        "Duplicate handling failed.",
                    )
                },
            )

        return {
            "current_df": result["dataframe"],
            **update_history(
                state,
                "duplicate_handling",
                "success",
                {
                    "duplicates_found": result["duplicate_count"],
                    "rows_removed": result["rows_removed"],
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "duplicate_handling", e)


# ==================================================
# 6. TYPE CORRECTION
# ==================================================

def type_correction_node(state: DataState) -> dict[str, Any]:
    try:
        conversions = state.get("type_conversions", {})

        if not conversions:
            return record_failure(
                state,
                "type_correction",
                ValueError(
                    "No type conversions were supplied. Specify the column "
                    "and requested type in your query."
                ),
            )

        result = type_correction(
            df=state["current_df"],
            conversions=conversions,
        )

        if not result.get("success", False):
            return update_history(
                state,
                "type_correction",
                "failed",
                {
                    "message": result.get(
                        "message",
                        "Type correction failed.",
                    )
                },
            )

        return {
            "current_df": result["dataframe"],
            **update_history(
                state,
                "type_correction",
                "success",
                {
                    "conversion_report": result.get(
                        "conversion_report", {}
                    )
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "type_correction", e)


# ==================================================
# 7. CHECK MISSING VALUES
# ==================================================

def check_missing_values_node(state: DataState) -> dict[str, Any]:
    try:
        df = state["current_df"]
        missing_counts = df.isna().sum()

        missing_columns = missing_counts[
            missing_counts > 0
        ].index.tolist()

        categorical_columns = [
            col
            for col in missing_columns
            if (
                not pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])
            )
        ]

        numerical_columns = [
            col
            for col in missing_columns
            if (
                pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_bool_dtype(df[col])
            )
        ]

        return {
            "missing_values_checked": True,
            "missing_columns": missing_columns,
            "categorical_missing_columns": categorical_columns,
            "numerical_missing_columns": numerical_columns,
            "missing_counts": {
                col: int(missing_counts[col])
                for col in missing_columns
            },
            "categorical_imputation_done": False,
            "bell_shape_checked": False,
            "outlier_detection_done": False,
            "skewness_checked": False,
            "outlier_median_done_columns": [],
            "capped_columns": [],
            "skewness_median_done_columns": [],
            "mean_done_columns": [],
            **update_history(
                state,
                "check_missing_values",
                "success",
                {
                    "missing_columns": missing_columns,
                    "categorical_missing_columns": (
                        categorical_columns
                    ),
                    "numerical_missing_columns": numerical_columns,
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "check_missing_values", e)


# ==================================================
# 8. HELPER FUNCTIONS
# ==================================================

def get_numerical_columns(state: DataState) -> list[str]:
    df = state["current_df"]
    selected = state.get("imputation_columns")

    if selected is None:
        selected = state.get("numerical_missing_columns", [])

    return [
        col
        for col in selected
        if (
            col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
        )
    ]


def get_outlier_columns(state: DataState) -> list[str]:
    report = state.get("outlier_report", {})
    results = report.get("results", {})

    return [
        col
        for col, details in results.items()
        if details.get("outlier_count", 0) > 0
    ]


def get_no_outlier_columns(state: DataState) -> list[str]:
    numerical_columns = state.get(
        "numerical_missing_columns", []
    )
    outlier_columns = set(get_outlier_columns(state))

    return [
        col
        for col in numerical_columns
        if col not in outlier_columns
    ]


def get_pending_columns(
    columns: list[str],
    processed_columns: list[str],
) -> list[str]:

    processed = set(processed_columns)

    return [
        col for col in columns
        if col not in processed
    ]


# ==================================================
# 9. MODE IMPUTATION
# ==================================================

def mode_imputation_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("categorical_missing_columns", [])

        if not columns:
            return {
                "categorical_imputation_done": True,
                **update_history(
                    state,
                    "mode_imputation",
                    "success",
                    {
                        "message": (
                            "No categorical missing values to impute."
                        )
                    },
                ),
            }

        result = mode_imputation(
            state["current_df"],
            columns=columns,
        )

        return {
            "current_df": result,
            "categorical_imputation_done": True,
            **update_history(
                state,
                "mode_imputation",
                "success",
                {
                    "strategy": "mode",
                    "columns": columns,
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "mode_imputation", e)


# ==================================================
# 10. CHECK BELL-SHAPED DISTRIBUTION
# ==================================================

def check_bell_shaped_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("numerical_missing_columns", [])

        if not columns:
            return {
                "bell_shape_checked": True,
                "bell_shaped_results": {},
                **update_history(
                    state,
                    "check_bell_shaped",
                    "success",
                    {
                        "message": (
                            "No numerical missing-value columns "
                            "to inspect."
                        )
                    },
                ),
            }

        result = check_bell_shaped(
            state["current_df"],
            columns=columns,
        )

        return {
            "bell_shape_checked": True,
            "bell_shaped_results": result,
            **update_history(
                state,
                "check_bell_shaped",
                "success",
                {"bell_shaped_results": result},
            ),
        }

    except Exception as e:
        return record_failure(state, "check_bell_shaped", e)


# ==================================================
# 11. DETECT OUTLIERS
# ==================================================

def detect_outliers_node(state: DataState) -> dict[str, Any]:
    try:
        df = state["current_df"]
        columns = state.get("numerical_missing_columns", [])
        bell_results = state.get("bell_shaped_results", {})
        bell_results = bell_results.get("results", bell_results)

        zscore_columns = [
            col
            for col in columns
            if (
                bell_results.get(col, {}).get("is_bell_shaped")
                is True
            )
        ]

        # Skewed, unknown, or inconclusive distributions use IQR.
        iqr_columns = [
            col for col in columns
            if col not in zscore_columns
        ]

        combined_results: dict[str, Any] = {}
        methods_used: dict[str, str] = {}

        if iqr_columns:
            report = detect_outliers(
                df,
                columns=iqr_columns,
                method="iqr",
            )

            for col, details in report["results"].items():
                combined_results[col] = details
                methods_used[col] = "iqr"

        if zscore_columns:
            report = detect_outliers(
                df,
                columns=zscore_columns,
                method="zscore",
            )

            for col, details in report["results"].items():
                combined_results[col] = details
                methods_used[col] = "zscore"

        result = {
            "methods_used": methods_used,
            "results": combined_results,
        }

        return {
            "outlier_detection_done": True,
            "outlier_report": result,
            **update_history(
                state,
                "detect_outliers",
                "success",
                {"outlier_report": result},
            ),
        }

    except Exception as e:
        return record_failure(state, "detect_outliers", e)


# ==================================================
# 12. MEDIAN IMPUTATION
# ==================================================

def median_imputation_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("imputation_columns", [])
        reason = state.get("imputation_reason", "")

        if not columns:
            return update_history(
                state,
                "median_imputation",
                "success",
                {
                    "message": (
                        "No columns selected for median imputation."
                    )
                },
            )

        result = median_imputation(
            state["current_df"],
            columns=columns,
        )

        updates: dict[str, Any] = {
            "current_df": result,
            **update_history(
                state,
                "median_imputation",
                "success",
                {
                    "strategy": "median",
                    "reason": reason,
                    "columns": columns,
                },
            ),
        }

        if reason == "outliers":
            done = state.get(
                "outlier_median_done_columns", []
            ).copy()

            done.extend(col for col in columns if col not in done)
            updates["outlier_median_done_columns"] = done

        elif reason == "skewed":
            done = state.get(
                "skewness_median_done_columns", []
            ).copy()

            done.extend(col for col in columns if col not in done)
            updates["skewness_median_done_columns"] = done

        return updates

    except Exception as e:
        return record_failure(state, "median_imputation", e)


# ==================================================
# 13. OUTLIER CAPPING
# ==================================================

def outlier_capping_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("imputation_columns", [])

        if not columns:
            return update_history(
                state,
                "outlier_capping",
                "success",
                {
                    "message": "No columns selected for outlier capping."
                },
            )

        median_done = set(
            state.get("outlier_median_done_columns", [])
        )
        missing_columns = set(state.get("numerical_missing_columns", []))
        columns_with_missing_values = set(columns).intersection(missing_columns)

        if not columns_with_missing_values.issubset(median_done):
            raise ValueError(
                "Outlier capping cannot run before median "
                "imputation for selected columns with missing values."
            )

        result = outlier_capping(
            state["current_df"],
            columns=columns,
        )

        capped = state.get("capped_columns", []).copy()
        capped.extend(col for col in columns if col not in capped)

        return {
            "current_df": result,
            "capped_columns": capped,
            **update_history(
                state,
                "outlier_capping",
                "success",
                {
                    "strategy": "IQR capping",
                    "columns": columns,
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "outlier_capping", e)


# ==================================================
# 14. CHECK SKEWNESS
# ==================================================

def check_skewness_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("imputation_columns", [])

        if not columns:
            return {
                "skewness_checked": True,
                "skewness_results": {},
                **update_history(
                    state,
                    "check_skewness",
                    "success",
                    {
                        "message": (
                            "No columns selected for skewness analysis."
                        )
                    },
                ),
            }

        result = check_skewness(
            state["current_df"],
            columns=columns,
        )

        return {
            "skewness_checked": True,
            "skewness_results": result,
            **update_history(
                state,
                "check_skewness",
                "success",
                {"skewness_results": result},
            ),
        }

    except Exception as e:
        return record_failure(state, "check_skewness", e)


# ==================================================
# 15. CHECK SYMMETRY
# ==================================================

def check_symmetry_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("imputation_columns", [])

        result = check_symmetry(
            state["current_df"],
            columns=columns,
        )

        return {
            "symmetry_results": result,
            **update_history(
                state,
                "check_symmetry",
                "success",
                {"symmetry_results": result},
            ),
        }

    except Exception as e:
        return record_failure(state, "check_symmetry", e)


# ==================================================
# 16. MEAN IMPUTATION
# ==================================================

def mean_imputation_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("imputation_columns", [])

        if not columns:
            return update_history(
                state,
                "mean_imputation",
                "success",
                {
                    "message": "No columns selected for mean imputation."
                },
            )

        result = mean_imputation(
            state["current_df"],
            columns=columns,
        )

        done = state.get("mean_done_columns", []).copy()
        done.extend(col for col in columns if col not in done)

        return {
            "current_df": result,
            "mean_done_columns": done,
            **update_history(
                state,
                "mean_imputation",
                "success",
                {
                    "strategy": "mean",
                    "columns": columns,
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "mean_imputation", e)


# ==================================================
# 17. ENCODING NODES
# ==================================================

def one_hot_encoding_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("encoding_columns", [])

        if not columns:
            return update_history(
                state,
                "one_hot_encoding",
                "success",
                {
                    "message": (
                        "No columns selected for one-hot encoding."
                    )
                },
            )

        result = one_hot_encoding(
            state["current_df"],
            columns=columns,
        )

        completed = state.get(
            "one_hot_encoded_columns", []
        ).copy()

        completed.extend(col for col in columns if col not in completed)

        return {
            "current_df": result,
            "one_hot_encoded_columns": completed,
            **update_history(
                state,
                "one_hot_encoding",
                "success",
                {
                    "method": "one-hot",
                    "columns": columns,
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "one_hot_encoding", e)


def ordinal_encoding_node(state: DataState) -> dict[str, Any]:
    try:
        mappings = state.get("ordinal_mappings", {})
        columns = state.get("encoding_columns", [])

        selected_mappings = {
            col: mappings[col]
            for col in columns
            if col in mappings
        }

        if not selected_mappings:
            return update_history(
                state,
                "ordinal_encoding",
                "success",
                {"message": "No ordinal columns selected."},
            )

        result = ordinal_encoding(
            state["current_df"],
            mappings=selected_mappings,
        )

        completed = state.get(
            "ordinal_encoded_columns", []
        ).copy()

        completed.extend(
            col for col in selected_mappings
            if col not in completed
        )

        return {
            "current_df": result,
            "ordinal_encoded_columns": completed,
            **update_history(
                state,
                "ordinal_encoding",
                "success",
                {
                    "method": "ordinal",
                    "columns": list(selected_mappings.keys()),
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "ordinal_encoding", e)


def frequency_encoding_node(state: DataState) -> dict[str, Any]:
    try:
        columns = state.get("encoding_columns", [])

        if not columns:
            return update_history(
                state,
                "frequency_encoding",
                "success",
                {
                    "message": (
                        "No columns selected for frequency encoding."
                    )
                },
            )

        result = frequency_encoding(
            state["current_df"],
            columns=columns,
        )

        completed = state.get(
            "frequency_encoded_columns", []
        ).copy()

        completed.extend(col for col in columns if col not in completed)

        return {
            "current_df": result,
            "frequency_encoded_columns": completed,
            **update_history(
                state,
                "frequency_encoding",
                "success",
                {
                    "method": "frequency",
                    "columns": columns,
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "frequency_encoding", e)


def encoding_plan_ready_node(state: DataState) -> dict[str, Any]:
    try:
        encoding_plan = build_encoding_plan(state)
        return {
            "encoding_plan": encoding_plan,
            "encoding_plan_ready": True,
            **update_history(
                state,
                "encoding_plan_ready",
                "success",
                {"encoding_plan": encoding_plan},
            ),
        }
    except Exception as e:
        return record_failure(state, "encoding_plan_ready", e)


# ==================================================
# 18. BUILD ENCODING PLAN
# ==================================================

def build_encoding_plan(
    state: DataState,
) -> dict[str, list[str]]:
    """
    Validate and return the per-column plan selected by the LLM.
    """

    df = state["current_df"]
    ordinal_mappings = state.get("ordinal_mappings", {})
    requested_columns = state.get("requested_encoding_columns", [])
    plan = state.get("requested_encoding_plan")

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    target_column = state.get("target_column")
    categorical_columns = [
        col for col in categorical_columns if col != target_column
    ]
    selected_columns = set(categorical_columns)
    if requested_columns:
        unknown_columns = set(requested_columns) - selected_columns
        if unknown_columns:
            raise ValueError(
                "Requested encoding columns must be categorical feature columns. "
                f"Invalid columns: {unknown_columns}."
            )
        selected_columns = set(requested_columns)

    if plan is None:
        # The non-agent graph has no LLM router, so retain its existing rule-based plan.
        threshold = int(state.get("high_cardinality_threshold", 10))
        if threshold < 2:
            raise ValueError("high_cardinality_threshold must be at least 2.")
        ordinal_columns = [
            column for column in selected_columns
            if column in ordinal_mappings
        ]
        frequency_columns = []
        one_hot_columns = []
        for column in selected_columns - set(ordinal_columns):
            if df[column].nunique(dropna=True) >= threshold:
                frequency_columns.append(column)
            else:
                one_hot_columns.append(column)
        plan = {
            "ordinal": ordinal_columns,
            "frequency": frequency_columns,
            "one_hot": one_hot_columns,
        }

    if not isinstance(plan, dict) or set(plan) != {
        "ordinal",
        "frequency",
        "one_hot",
    }:
        raise ValueError(
            "The LLM must provide an encoding plan with ordinal, frequency, "
            "and one_hot column lists."
        )
    if not all(
        isinstance(columns, list)
        and all(isinstance(column, str) for column in columns)
        for columns in plan.values()
    ):
        raise ValueError("Every encoding-plan strategy must contain a list of column names.")

    planned_columns = (
        plan["ordinal"] + plan["frequency"] + plan["one_hot"]
    )
    if len(planned_columns) != len(set(planned_columns)):
        raise ValueError(
            "Each column must appear in exactly one encoding-plan strategy."
        )
    if set(planned_columns) != selected_columns:
        missing = selected_columns - set(planned_columns)
        unexpected = set(planned_columns) - selected_columns
        raise ValueError(
            "The LLM encoding plan must cover exactly the selected categorical "
            f"features. Missing: {missing}; unexpected: {unexpected}."
        )

    for col in plan["ordinal"]:
        if col not in ordinal_mappings:
            raise ValueError(
                f"Ordinal encoding for '{col}' requires an explicit category mapping."
            )
        mapping = ordinal_mappings[col]
        observed = set(df[col].dropna().unique())
        mapped = set(mapping)
        if observed != mapped:
            missing = observed - mapped
            extra = mapped - observed
            raise ValueError(
                f"Ordinal mapping for '{col}' must cover exactly the observed "
                f"categories. Unmapped: {missing}; unobserved: {extra}."
            )

    return {
        "ordinal": plan["ordinal"],
        "frequency": plan["frequency"],
        "one_hot": plan["one_hot"],
    }


# ==================================================
# 19. SCALING NODES
# ==================================================

def standard_scaling_node(state: DataState) -> dict[str, Any]:
    try:
        result = standard_scaling(state["current_df"])

        return {
            "current_df": result,
            **update_history(
                state,
                "standard_scaling",
                "success",
            ),
        }

    except Exception as e:
        return record_failure(state, "standard_scaling", e)


def minmax_scaling_node(state: DataState) -> dict[str, Any]:
    try:
        result = minmax_scaling(state["current_df"])

        return {
            "current_df": result,
            **update_history(
                state,
                "minmax_scaling",
                "success",
            ),
        }

    except Exception as e:
        return record_failure(state, "minmax_scaling", e)


def robust_scaling_node(state: DataState) -> dict[str, Any]:
    try:
        result = robust_scaling(state["current_df"])

        return {
            "current_df": result,
            **update_history(
                state,
                "robust_scaling",
                "success",
            ),
        }

    except Exception as e:
        return record_failure(state, "robust_scaling", e)


# ==================================================
# 20. FEATURE SCALING NODE
# ==================================================

def feature_scaling_node(state: DataState) -> dict[str, Any]:
    """
    Optional final scaling stage.

    scaling_method:
        None       -> skip scaling
        "standard" -> StandardScaler
        "minmax"   -> MinMaxScaler
        "robust"   -> RobustScaler

    Scales numerical feature columns identified during inspection,
    excluding the target column and Boolean columns.
    """

    try:
        method = state.get("scaling_method")

        if not method:
            return {
                "feature_scaling_done": True,
                "scaling_columns": [],
                **update_history(
                    state,
                    "feature_scaling",
                    "success",
                    {"message": "Feature scaling skipped."},
                ),
            }

        method = str(method).strip().lower()

        scalers = {
            "standard": standard_scaling,
            "minmax": minmax_scaling,
            "robust": robust_scaling,
        }

        if method not in scalers:
            raise ValueError(
                "Invalid scaling_method. Choose "
                "'standard', 'minmax', or 'robust'."
            )

        df = state["current_df"].copy()
        inspection = state.get("inspection_results", {})

        original_numerical_columns = inspection.get(
            "numerical_columns", []
        )
        if not original_numerical_columns:
            original_numerical_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

        target_column = state.get("target_column")

        scaling_columns = [
            col
            for col in original_numerical_columns
            if (
                col in df.columns
                and col != target_column
                and pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_bool_dtype(df[col])
            )
        ]

        if not scaling_columns:
            return {
                "feature_scaling_done": True,
                "scaling_columns": [],
                **update_history(
                    state,
                    "feature_scaling",
                    "success",
                    {
                        "message": (
                            "No numerical feature columns "
                            "available for scaling."
                        )
                    },
                ),
            }

        # The scaling functions must accept a DataFrame and return
        # the transformed DataFrame with the same columns.
        scaled_data = scalers[method](
            df[scaling_columns].copy()
        )

        df.loc[:, scaling_columns] = scaled_data

        return {
            "current_df": df,
            "feature_scaling_done": True,
            "scaling_method_used": method,
            "scaling_columns": scaling_columns,
            **update_history(
                state,
                "feature_scaling",
                "success",
                {
                    "method": method,
                    "columns": scaling_columns,
                },
            ),
        }

    except Exception as e:
        return record_failure(state, "feature_scaling", e)


# ==================================================
# 21. DECISION NODE
# ==================================================

def decision_node(state: DataState) -> dict[str, Any]:
    """
    Workflow:
    1. Check missing values.
    2. Categorical missing values -> mode imputation.
    3. Numerical missing values:
       - Bell-shaped -> Z-score outlier detection.
       - Otherwise -> IQR outlier detection.
       - Outliers -> median imputation -> capping.
       - No outliers -> skewness check -> median or mean.
    4. Encoding:
       - Ordinal.
       - Frequency.
       - One-hot.
    5. Optional feature scaling.
    6. Finish.
    """

    def choose(
        action: str,
        reason: str,
        **extra: Any,
    ) -> dict[str, Any]:

        return {
            **extra,
            "next_action": action,
            **update_history(
                state,
                "decision",
                "success",
                {
                    "next_action": action,
                    "reason": reason,
                },
            ),
        }

    # A. Stop safely if inspection failed.
    inspection = state.get("inspection_results", {})

    if inspection and not inspection.get("success", False):
        return choose(
            "finish",
            "Dataset inspection failed; stopping safely.",
        )

    # B. Check missing values.
    if not state.get("missing_values_checked", False):
        return choose(
            "check_missing_values",
            "Check missing values after type correction.",
        )

    # C. Categorical missing values -> mode.
    categorical_columns = state.get(
        "categorical_missing_columns", []
    )

    if (
        categorical_columns
        and not state.get("categorical_imputation_done", False)
    ):
        return choose(
            "mode_imputation",
            "Categorical columns contain missing values.",
        )

    # D. Numerical missing-value processing.
    numerical_columns = state.get(
        "numerical_missing_columns", []
    )

    if numerical_columns:

        # D1. Check distribution shape.
        if not state.get("bell_shape_checked", False):
            return choose(
                "check_bell_shaped",
                (
                    "Check numerical distributions before "
                    "choosing an outlier detection method."
                ),
            )

        # D2. Detect outliers.
        if not state.get("outlier_detection_done", False):
            return choose(
                "detect_outliers",
                "Detect outliers using the selected method.",
            )

        outlier_columns = get_outlier_columns(state)

        # D3. Outliers -> median imputation -> capping.
        if outlier_columns:
            median_done = state.get(
                "outlier_median_done_columns", []
            )

            pending_median = get_pending_columns(
                outlier_columns,
                median_done,
            )

            if pending_median:
                return choose(
                    "median_imputation",
                    (
                        "Impute columns containing outliers "
                        "with the median."
                    ),
                    imputation_columns=pending_median,
                    imputation_reason="outliers",
                )

            capped_columns = state.get("capped_columns", [])

            pending_capping = get_pending_columns(
                outlier_columns,
                capped_columns,
            )

            if pending_capping:
                return choose(
                    "outlier_capping",
                    (
                        "Median imputation is complete; "
                        "cap the detected outliers."
                    ),
                    imputation_columns=pending_capping,
                )

        # D4. No-outlier columns -> skewness check.
        no_outlier_columns = get_no_outlier_columns(state)

        if (
            no_outlier_columns
            and not state.get("skewness_checked", False)
        ):
            return choose(
                "check_skewness",
                (
                    "No outliers were detected in these columns; "
                    "check their skewness."
                ),
                imputation_columns=no_outlier_columns,
            )

        # D5. Interpret skewness results.
        skewness_results = state.get("skewness_results", {})

        def column_is_skewed(col: str) -> bool:
            details = skewness_results.get(col, {})

            if isinstance(details, dict):
                if "is_skewed" in details:
                    return bool(details["is_skewed"])

                value = details.get("skewness")

                if value is not None:
                    return abs(float(value)) > 0.5

            if isinstance(details, (int, float)):
                return abs(float(details)) > 0.5

            return False

        median_done = state.get(
            "skewness_median_done_columns", []
        )

        mean_done = state.get("mean_done_columns", [])

        skewed_columns = [
            col
            for col in no_outlier_columns
            if column_is_skewed(col)
        ]

        symmetric_columns = [
            col
            for col in no_outlier_columns
            if not column_is_skewed(col)
        ]

        # D6. Skewed columns -> median.
        pending_skewed = get_pending_columns(
            skewed_columns,
            median_done,
        )

        if pending_skewed:
            return choose(
                "median_imputation",
                (
                    "Columns without outliers are skewed; "
                    "use median imputation."
                ),
                imputation_columns=pending_skewed,
                imputation_reason="skewed",
            )

        # D7. Approximately symmetric columns -> mean.
        pending_symmetric = get_pending_columns(
            symmetric_columns,
            mean_done,
        )

        if pending_symmetric:
            return choose(
                "mean_imputation",
                (
                    "Columns without outliers are approximately "
                    "symmetric; use mean imputation."
                ),
                imputation_columns=pending_symmetric,
            )

    # E. Build encoding plan.
    encoding_plan = state.get("encoding_plan")

    if encoding_plan is None:
        try:
            encoding_plan = build_encoding_plan(state)

        except Exception as e:
            return {
                "next_action": "finish",
                "workflow_error": str(e),
                **record_failure(
                    state,
                    "build_encoding_plan",
                    e,
                ),
            }

        return choose(
            "encoding_plan_ready",
            "Encoding plan created.",
            encoding_plan=encoding_plan,
        )

    # F. Ordinal encoding.
    ordinal_columns = encoding_plan.get("ordinal", [])

    pending_ordinal = get_pending_columns(
        ordinal_columns,
        state.get("ordinal_encoded_columns", []),
    )

    if pending_ordinal:
        return choose(
            "ordinal_encoding",
            "Encode ordinal columns using supplied mappings.",
            encoding_columns=pending_ordinal,
        )

    # G. Frequency encoding.
    frequency_columns = encoding_plan.get("frequency", [])

    pending_frequency = get_pending_columns(
        frequency_columns,
        state.get("frequency_encoded_columns", []),
    )

    if pending_frequency:
        return choose(
            "frequency_encoding",
            "Encode high-cardinality categorical columns.",
            encoding_columns=pending_frequency,
        )

    # H. One-hot encoding.
    one_hot_columns = encoding_plan.get("one_hot", [])

    pending_one_hot = get_pending_columns(
        one_hot_columns,
        state.get("one_hot_encoded_columns", []),
    )

    if pending_one_hot:
        return choose(
            "one_hot_encoding",
            "Encode low-cardinality categorical columns.",
            encoding_columns=pending_one_hot,
        )

    # I. Optional feature scaling.
    scaling_method = state.get("scaling_method")

    if (
        scaling_method
        and not state.get("feature_scaling_done", False)
    ):
        return choose(
            "feature_scaling",
            "Apply the requested feature scaling method.",
        )

    # J. Finish.
    return choose(
        "finish",
        (
            "Missing-value handling, outlier processing, "
            "categorical encoding, and optional feature "
            "scaling are complete."
        ),
    )


# ==================================================
# 22. FINISH NODE
# ==================================================

def finish_node(state: DataState) -> dict[str, Any]:
    return {
        "finished": True,
        "next_action": "finish",
        **update_history(
            state,
            "finish",
            "success",
            {
                "original_shape": state["original_df"].shape,
                "final_shape": state["current_df"].shape,
                "completed_steps": state.get(
                    "completed_steps", []
                ).copy(),
            },
        ),
    }