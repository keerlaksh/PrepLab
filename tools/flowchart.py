
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config


NODES = [
    ("problem_definition", "1. Problem Definition", "setup"),
    ("inspect_data", "2. Inspect Data", "setup"),
    ("clean_columns", "3. Clean Columns", "setup"),
    ("duplicate_handling", "4. Handle Duplicates", "setup"),
    ("type_correction", "5. Correct Data Types", "setup"),
    ("check_missing_values", "6. Check Missing Values", "decision"),
    ("no_missing", "No Missing Values", "success"),
    ("identify_missing", "Identify Missing Columns", "process"),
    ("categorical", "Categorical Columns", "decision"),
    ("numerical", "Numerical Columns", "decision"),
    ("mode_imputation", "Mode Imputation", "process"),
    ("choose_method", "Choose Detection Method", "decision"),
    ("iqr", "IQR Detection", "process"),
    ("zscore", "Z-Score Detection", "process"),
    ("detect_outliers", "Detect Outliers", "decision"),
    ("outliers_found", "Outliers Found", "decision"),
    ("no_outliers", "No Outliers", "decision"),
    ("median_imputation", "Median Imputation", "process"),
    ("outlier_capping", "Cap Outliers", "process"),
    ("check_skewness", "Check Skewness", "decision"),
    ("skewed", "Skewed", "decision"),
    ("symmetric", "Approximately Symmetric", "decision"),
    ("mean_imputation", "Mean Imputation", "process"),
    ("decision_node", "Decision Node", "decision"),
    ("ordinal_encoding", "Ordinal Encoding", "process"),
    ("frequency_encoding", "Frequency Encoding", "process"),
    ("one_hot_encoding", "One-Hot Encoding", "process"),
    ("scaling", "Optional Feature Scaling", "process"),
    ("finish", "Preprocessing Complete", "success"),
]

# Every edge is explicitly defined so the connections are visible.
EDGES = [
    ("problem_definition", "inspect_data", ""),
    ("inspect_data", "clean_columns", ""),
    ("clean_columns", "duplicate_handling", ""),
    ("duplicate_handling", "type_correction", ""),
    ("type_correction", "check_missing_values", ""),
    ("check_missing_values", "no_missing", "No"),
    ("check_missing_values", "identify_missing", "Yes"),
    ("identify_missing", "categorical", "Categorical"),
    ("identify_missing", "numerical", "Numerical"),
    ("categorical", "mode_imputation", ""),
    ("mode_imputation", "decision_node", ""),
    ("no_missing", "decision_node", ""),
    ("numerical", "choose_method", ""),
    ("choose_method", "iqr", "Skewed / unknown"),
    ("choose_method", "zscore", "Bell-shaped"),
    ("iqr", "detect_outliers", ""),
    ("zscore", "detect_outliers", ""),
    ("detect_outliers", "outliers_found", "Yes"),
    ("detect_outliers", "no_outliers", "No"),
    ("outliers_found", "median_imputation", ""),
    ("median_imputation", "outlier_capping", ""),
    ("outlier_capping", "decision_node", ""),
    ("no_outliers", "check_skewness", ""),
    ("check_skewness", "skewed", "Yes"),
    ("check_skewness", "symmetric", "No"),
    ("skewed", "median_imputation", ""),
    ("symmetric", "mean_imputation", ""),
    ("mean_imputation", "decision_node", ""),
    ("decision_node", "ordinal_encoding", "If mappings supplied"),
    ("ordinal_encoding", "frequency_encoding", ""),
    ("frequency_encoding", "one_hot_encoding", ""),
    ("one_hot_encoding", "scaling", ""),
    ("scaling", "finish", ""),
]

NODE_DETAILS = {
    "problem_definition": "Determine the task, target column, and preprocessing objective.",
    "inspect_data": "Inspect shape, column types, missing values, and cardinality.",
    "clean_columns": "Clean column names and identify columns requiring attention.",
    "duplicate_handling": "Check and handle duplicate rows.",
    "type_correction": "Correct column data types where appropriate.",
    "check_missing_values": "Check whether missing values remain.",
    "no_missing": "No missing values were found. Continue to the decision workflow.",
    "identify_missing": "Separate columns with missing values by data type.",
    "categorical": "Categorical columns follow the mode-imputation path.",
    "numerical": "Numerical columns proceed to detection-method selection.",
    "mode_imputation": "Fill missing categorical values using the mode.",
    "choose_method": "Choose IQR or Z-score based on the distribution.",
    "iqr": "Use the interquartile range rule for outlier detection.",
    "zscore": "Use standardized distance from the mean to detect outliers.",
    "detect_outliers": "Determine whether the selected method detects outliers.",
    "outliers_found": "The outlier branch leads to median imputation and capping.",
    "no_outliers": "Continue to skewness analysis when no outliers are detected.",
    "median_imputation": "Fill applicable missing numerical values with the median.",
    "outlier_capping": "Cap detected outliers rather than deleting rows.",
    "check_skewness": "Check the numerical distribution's skewness.",
    "skewed": "The proposed path uses median imputation.",
    "symmetric": "The proposed path uses mean imputation.",
    "mean_imputation": "Fill applicable missing numerical values with the mean.",
    "decision_node": "The workflow decision node chooses the next action.",
    "ordinal_encoding": "Apply explicitly supplied ordinal mappings.",
    "frequency_encoding": "Encode applicable high-cardinality categorical columns.",
    "one_hot_encoding": "One-hot encode applicable categorical columns.",
    "scaling": "Apply the selected scaling method, if enabled.",
    "finish": "The workflow has reached its final stage.",
}

STATUS_COLORS = {
    "setup": "#DCEAFE",
    "decision": "#FFF0C2",
    "process": "#E2F5E9",
    "success": "#D8F3F0",
}


def show_flowchart(processing_history=None, selected_node=None):
    st.subheader("Preprocessing Workflow")
    st.caption(
        "Zoom with the mouse wheel or trackpad. Drag the canvas to pan. "
        "Select a node to inspect its details."
    )

    history = processing_history or []
    history_by_step = {
        str(item.get("step", "")): item for item in history
    }

    node_objects = []
    for node_id, label, category in NODES:
        entry = history_by_step.get(node_id, {})
        status = entry.get("status", "pending")
        color = STATUS_COLORS[category]

        if status == "success":
            color = "#A8E6B0"
        elif status == "failed":
            color = "#FFB3B3"

        node_objects.append(
            Node(
                id=node_id,
                label=f"{label}\n[{status.upper()}]",
                title=NODE_DETAILS.get(node_id, label),
                size=28,
                color=color,
                shape="box",
                font={"size": 20, "face": "Arial"},
                borderWidth=2,
            )
        )

    edge_objects = [
        Edge(
            source=source,
            target=target,
            label=label,
            color="#64748B",
            width=2,
            font={"size": 15},
            arrows="to",
        )
        for source, target, label in EDGES
    ]

    config = Config(
        width=1500,
        height=1150,
        directed=True,
        physics=False,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F59E0B",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": True},
        layout={
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 130,
                "nodeSpacing": 220,
                "treeSpacing": 260,
            }
        },
    )

    clicked = agraph(
        nodes=node_objects,
        edges=edge_objects,
        config=config,
    )

    node_ids = [node[0] for node in NODES]
    labels = {node_id: label for node_id, label, _ in NODES}

    if clicked and getattr(clicked, "id", None) in labels:
        st.session_state["flow_selected_node"] = clicked.id

    chosen = st.session_state.get("flow_selected_node", selected_node)
    if chosen not in labels:
        chosen = node_ids[0]

    st.divider()
    st.subheader("Selected Node Details")

    chosen_label = labels[chosen]
    st.markdown(f"### {chosen_label}")
    st.write(NODE_DETAILS.get(chosen, "No description available."))

    entry = history_by_step.get(chosen)
    if entry:
        st.write("**Execution status:**", entry.get("status", "unknown"))
        st.json(entry.get("details", {}))
    else:
        st.info(
            "This node has no matching execution-history entry yet. "
            "Its status is pending or it may be a conceptual branch."
        )

    incoming = [source for source, target, _ in EDGES if target == chosen]
    outgoing = [target for source, target, _ in EDGES if source == chosen]

    st.write("**Connected from:**")
    st.write([labels[x] for x in incoming] or ["Start of diagram"])

    st.write("**Connects to:**")
    st.write([labels[x] for x in outgoing] or ["End of diagram"])

    selected = st.selectbox(
        "Or choose a node to inspect",
        options=node_ids,
        index=node_ids.index(chosen),
        format_func=lambda x: labels[x],
        key="flow_node_dropdown",
    )
    if selected != chosen:
        st.session_state["flow_selected_node"] = selected
        st.rerun()
