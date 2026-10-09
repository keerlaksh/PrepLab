Markdown


# Step 1: Problem Definition Tool

- Created `problem_definition.py` to analyze an uploaded dataset.
- Extracts dataset information, including shape, column names, data types, missing values, and duplicate rows.
- Generates a statistical summary and samples the first five rows.
- Uses LLaMA 3.3 via Groq to identify the likely machine learning problem (classification, regression, clustering, etc.).
- Suggests a possible target column and identifies potential data-quality issues.
- Recommends the next steps required before model training.
- Returns the analysis and dataset metadata as a Python dictionary.
- Does not modify the original dataset.



## Step 2: Data Inspection Tool

- Created `inspect_data.py` to inspect the dataset's structural characteristics.
- Detects duplicate rows using `df.duplicated().sum()`.
- Identifies categorical columns with `object` dtype.
- Identifies columns with dtype other than `object`, following the current implementation.
- Calculates categorical cardinality using `df[col].nunique()`.
- Returns inspection results as a Python dictionary for subsequent tools.
- Does not modify the dataset or remove duplicates.
- Does not repeat Step 1's LLM-based problem definition or target-column analysis.

