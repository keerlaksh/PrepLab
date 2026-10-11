
from langgraph.graph import StateGraph, START, END

from agent.state import DataState

from agent.nodes import (
    # Initial preprocessing
    problem_definition_node,
    inspect_data_node,
    clean_columns_node,
    duplicate_handling_node,
    type_correction_node,

    # Missing values
    check_missing_values_node,

    # Imputation
    mode_imputation_node,
    mean_imputation_node,
    median_imputation_node,

    # Distribution analysis
    check_skewness_node,
    check_symmetry_node,
    check_bell_shaped_node,

    # Outliers
    detect_outliers_node,
    outlier_capping_node,

    # Encoding
    ordinal_encoding_node,
    frequency_encoding_node,
    one_hot_encoding_node,

    # Feature scaling
    feature_scaling_node,

    # Decision and finish
    decision_node,
    encoding_plan_ready_node,
    finish_node,
)


# ==================================================
# 1. ROUTE BASED ON DECISION NODE
# ==================================================

def route_next_action(state: DataState) -> str:

    allowed_actions = {
        # Missing values
        "check_missing_values",

        # Imputation
        "mode_imputation",
        "mean_imputation",
        "median_imputation",

        # Distribution analysis
        "check_bell_shaped",
        "check_skewness",
        "check_symmetry",

        # Outlier processing
        "detect_outliers",
        "outlier_capping",

        # Encoding
        "encoding_plan_ready",
        "ordinal_encoding",
        "frequency_encoding",
        "one_hot_encoding",

        # Optional scaling
        "feature_scaling",

        # Finish
        "finish",
    }

    action = state.get("next_action", "finish")

    if action not in allowed_actions:
        return "finish"

    return action


# ==================================================
# 2. BUILD PREPROCESSING GRAPH
# ==================================================

def build_graph():

    graph = StateGraph(DataState)

    # ----------------------------------------------
    # A. Initial preprocessing
    # ----------------------------------------------

    graph.add_node(
        "problem_definition",
        problem_definition_node,
    )

    graph.add_node(
        "inspect_data",
        inspect_data_node,
    )

    graph.add_node(
        "clean_columns",
        clean_columns_node,
    )

    graph.add_node(
        "duplicate_handling",
        duplicate_handling_node,
    )

    graph.add_node(
        "type_correction",
        type_correction_node,
    )

    # ----------------------------------------------
    # B. Missing-value detection
    # ----------------------------------------------

    graph.add_node(
        "check_missing_values",
        check_missing_values_node,
    )

    # ----------------------------------------------
    # C. Imputation
    # ----------------------------------------------

    graph.add_node(
        "mode_imputation",
        mode_imputation_node,
    )

    graph.add_node(
        "mean_imputation",
        mean_imputation_node,
    )

    graph.add_node(
        "median_imputation",
        median_imputation_node,
    )

    # ----------------------------------------------
    # D. Distribution analysis
    # ----------------------------------------------

    graph.add_node(
        "check_bell_shaped",
        check_bell_shaped_node,
    )

    graph.add_node(
        "check_skewness",
        check_skewness_node,
    )

    graph.add_node(
        "check_symmetry",
        check_symmetry_node,
    )

    # ----------------------------------------------
    # E. Outlier processing
    # ----------------------------------------------

    graph.add_node(
        "detect_outliers",
        detect_outliers_node,
    )

    graph.add_node(
        "outlier_capping",
        outlier_capping_node,
    )

    # ----------------------------------------------
    # F. Encoding
    # ----------------------------------------------

    graph.add_node(
        "ordinal_encoding",
        ordinal_encoding_node,
    )

    graph.add_node(
        "frequency_encoding",
        frequency_encoding_node,
    )

    graph.add_node(
        "one_hot_encoding",
        one_hot_encoding_node,
    )

    graph.add_node(
        "encoding_plan_ready",
        encoding_plan_ready_node,
    )

    # ----------------------------------------------
    # G. Optional feature scaling
    # ----------------------------------------------

    graph.add_node(
        "feature_scaling",
        feature_scaling_node,
    )

    # ----------------------------------------------
    # H. Decision and finish
    # ----------------------------------------------

    graph.add_node(
        "decision",
        decision_node,
    )

    graph.add_node(
        "finish",
        finish_node,
    )

    # ==================================================
    # 3. INITIAL WORKFLOW
    # ==================================================

    graph.add_edge(
        START,
        "problem_definition",
    )

    graph.add_edge(
        "problem_definition",
        "inspect_data",
    )

    graph.add_edge(
        "inspect_data",
        "clean_columns",
    )

    graph.add_edge(
        "clean_columns",
        "duplicate_handling",
    )

    graph.add_edge(
        "duplicate_handling",
        "type_correction",
    )

    graph.add_edge(
        "type_correction",
        "decision",
    )

    # ==================================================
    # 4. CONDITIONAL ROUTING FROM DECISION
    # ==================================================

    graph.add_conditional_edges(
        "decision",
        route_next_action,
        {
            "check_missing_values": "check_missing_values",

            "mode_imputation": "mode_imputation",
            "mean_imputation": "mean_imputation",
            "median_imputation": "median_imputation",

            "check_bell_shaped": "check_bell_shaped",
            "check_skewness": "check_skewness",
            "check_symmetry": "check_symmetry",

            "detect_outliers": "detect_outliers",
            "outlier_capping": "outlier_capping",

            "encoding_plan_ready": "encoding_plan_ready",
            "ordinal_encoding": "ordinal_encoding",
            "frequency_encoding": "frequency_encoding",
            "one_hot_encoding": "one_hot_encoding",

            "feature_scaling": "feature_scaling",

            "finish": "finish",
        },
    )

    # ==================================================
    # 5. RETURN PROCESSING NODES TO DECISION
    # ==================================================

    processing_nodes = [
        "check_missing_values",
        "mode_imputation",
        "mean_imputation",
        "median_imputation",
        "check_bell_shaped",
        "check_skewness",
        "check_symmetry",
        "detect_outliers",
        "outlier_capping",
        "ordinal_encoding",
        "frequency_encoding",
        "one_hot_encoding",
        "encoding_plan_ready",
        "feature_scaling",
    ]

    for node_name in processing_nodes:
        graph.add_edge(
            node_name,
            "decision",
        )

    # ==================================================
    # 6. FINISH
    # ==================================================

    graph.add_edge(
        "finish",
        END,
    )

    return graph.compile()


# ==================================================
# 7. COMPILED GRAPH
# ==================================================

preprocessing_graph = build_graph()

# Keep compatibility with code using the original name.
build_preprocessing_graph = build_graph