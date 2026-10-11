
from typing import Any, TypedDict
import pandas as pd


class DataState(TypedDict, total=False):
    # --------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------
    current_df: pd.DataFrame
    original_df: pd.DataFrame
    file_path: str

    # --------------------------------------------------
    # PROBLEM DEFINITION
    # --------------------------------------------------
    problem_type: str
    target_column: str
    problem_definition: dict[str, Any]
    problem_summary: str
    type_conversions: dict[str, str]
    requested_encoding_columns: list[str]
    requested_encoding_plan: dict[str, list[str]]

    # --------------------------------------------------
    # DATA INSPECTION
    # --------------------------------------------------
    inspection_results: dict[str, Any]
    original_shape: tuple[int, int]
    final_shape: tuple[int, int]

    # --------------------------------------------------
    # PREPROCESSING STATUS
    # --------------------------------------------------
    completed_steps: list[str]
    processing_history: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]

    # --------------------------------------------------
    # MISSING VALUE CHECKING
    # --------------------------------------------------
    missing_values_checked: bool
    missing_values_report: dict[str, Any]

    categorical_missing_columns: list[str]
    numerical_missing_columns: list[str]

    categorical_imputation_done: bool

    # --------------------------------------------------
    # DISTRIBUTION AND OUTLIER DETECTION
    # --------------------------------------------------
    bell_shape_checked: bool
    bell_shape_results: dict[str, Any]
    is_bell_shaped: bool

    outlier_detection_done: bool
    outlier_report: dict[str, Any]
    outlier_columns: list[str]
    no_outlier_columns: list[str]

    # --------------------------------------------------
    # IMPUTATION TRACKING
    # --------------------------------------------------
    imputation_columns: list[str]
    imputation_reason: str

    outlier_median_done_columns: list[str]
    capped_columns: list[str]

    skewness_checked: bool
    skewness_results: dict[str, Any]
    skewness_median_done_columns: list[str]
    mean_done_columns: list[str]

    # --------------------------------------------------
    # ENCODING CONFIGURATION
    # --------------------------------------------------
    encoding_columns: list[str]
    ordinal_mappings: dict[str, dict[str, Any]]
    high_cardinality_threshold: int

    encoding_plan: dict[str, list[str]]
    encoding_plan_ready: bool

    ordinal_encoded_columns: list[str]
    frequency_encoded_columns: list[str]
    one_hot_encoded_columns: list[str]

    # --------------------------------------------------
    # OPTIONAL FEATURE SCALING
    # --------------------------------------------------
    scaling_method: str | None
    scaling_method_used: str | None
    scaling_columns: list[str]
    feature_scaling_done: bool

    # --------------------------------------------------
    # GRAPH CONTROL
    # --------------------------------------------------
    next_action: str
    requested_workflow: str
    finished: bool