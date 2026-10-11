# PrepLab

PrepLab is a Streamlit application for preprocessing tabular datasets through plain-language requests. It uses an LLM to select relevant preprocessing tools, executes those tools through LangGraph, displays the actual route taken, and provides downloadable processed data and a run report.

## Features

- Upload and preview CSV datasets.
- Describe a preprocessing task in natural language.
- Route each step through an LLM using the user’s request, dataset profile, available tools, workflow rules, and previous results.
- Run individual tools or relevant branches rather than automatically applying every preprocessing step.
- Display the executed route in the application’s flowchart.
- Download the processed dataset as a CSV and the run details as a JSON report.

## How It Works

The main application is `tools/app.py`. It reads the uploaded CSV into a pandas DataFrame and passes that dataset and the user’s request to the Agent workflow.

The Agent workflow is implemented in `tools/agent/agentic.py`:

1. The router builds a profile of the current dataset, including column names, data types, categorical values, cardinalities, missing-value counts, and the selected target column.
2. It sends the user request, dataset profile, previously executed steps, and currently available tools to the configured Groq model.
3. The LLM selects one next tool at a time.
4. PrepLab checks whether that tool is registered, allowed, and ready to run. Invalid choices are rejected and the LLM is asked to choose again.
5. The selected tool runs, and its result becomes part of the context for the next routing decision.
6. The workflow finishes when the LLM selects `finish` and the requested workflow’s completion checks are satisfied.

The application records routing choices and processing details for the live flowchart and downloadable report.

> **Current application behavior:** The Streamlit interface sets `run_mode` to Agent mode. A separate graph implementation exists in `tools/agent/graph.py`, but the current interface does not expose a mode selector for it.

## Available Preprocessing Workflows

### Dataset analysis

- **Problem definition:** Sends dataset statistics and sample rows to an LLM for an analysis of the likely task, possible target, data-quality concerns, and recommended next steps. It does not modify the DataFrame.
- **Data inspection:** Reports duplicate-row count, categorical and numerical columns, and categorical cardinalities.
- **Column cleaning:** Cleans or reviews column names and columns using the project’s column-cleaning implementation.
- **Duplicate handling:** Detects duplicate rows and removes them when invoked by the Agent workflow.
- **Type correction:** Converts specified columns to a requested type. Supported types include numeric, integer, float, string, category, datetime, and boolean. Invalid conversions may become missing values and are included in the conversion report.

### Missing values and outliers

The Agent’s missing-value guidance distinguishes categorical and numerical columns:

- Categorical missing values can be filled with the mode.
- Numerical missing values can be assessed for distribution shape and outliers.
- Approximately bell-shaped numerical columns use Z-score outlier detection; other or inconclusive distributions use IQR.
- For the imputation workflow, numerical columns with outliers use median imputation followed by outlier capping.
- Numerical columns without outliers are assessed for skewness: skewed columns use median imputation, while approximately symmetric columns use mean imputation.
- A check-only request can report missing values without performing imputation.
- Outlier capping is performed only when the requested workflow includes capping.

### Categorical encoding

The encoding tools are:

- **Ordinal encoding:** Maps category values to numbers using a supplied category-to-number mapping. The mapping must cover the column’s observed non-missing categories.
- **Frequency encoding:** Replaces category values with their normalized frequency in that column.
- **One-hot encoding:** Creates indicator columns for the selected categories.

For an encoding request, the LLM receives eligible categorical columns, their unique counts, sample values, the high-cardinality threshold, and the user’s query. It returns a per-column plan assigning each selected column to an encoding method. The plan is checked for valid structure, coverage, duplicate assignments, and valid ordinal mappings before it runs.

The workflow rules guide the LLM’s choice: ordinal for clearly ordered categories with mappings, frequency for high-cardinality nominal features, and one-hot for low-cardinality nominal features. A specific method requested by the user can guide the plan, subject to validation.

### Feature scaling

PrepLab supports:

- Standard scaling
- Min-max scaling
- Robust scaling

The scaling workflow excludes the selected target column and Boolean columns. The scaling tools require missing values in the selected numerical columns to be handled first.

## Project Structure

```text
PrepLab/
├── .gitignore
└── tools/
    ├── app.py
    ├── clean_columns.py
    ├── inspect_data.py
    ├── problem_definition.py
    ├── flowchart.py                 # Standalone/legacy chart; not imported by app.py
    ├── agent/
    │   ├── __init__.py
    │   ├── agentic.py               # LLM router and Agent graph
    │   ├── graph.py                 # LangGraph workflow definition
    │   ├── nodes.py                 # Processing-node implementations
    │   └── state.py                 # Shared LangGraph state schema
    ├── data_quality/
    │   ├── duplicate_handling.py
    │   └── type_correction.py
    ├── encoding/
    │   ├── frequency_encoding.py
    │   ├── one_hot_encoding.py
    │   └── ordinal_encoding.py
    ├── imputation/
    │   ├── check_skewness.py
    │   ├── check_symmetry.py
    │   ├── mean_imputation.py
    │   ├── median_imputation.py
    │   └── mode_imputation.py
    ├── outliers/
    │   ├── check_bell_shaped.py
    │   ├── detect_outliers.py
    │   ├── iqr_strategy.py
    │   ├── outlier_capping.py
    │   └── zscore_strategy.py
    └── scaling/
        ├── minmax_scaling.py
        ├── robust_scaling.py
        └── standard_scaling.py
