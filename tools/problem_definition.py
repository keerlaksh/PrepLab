
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


# --------------------------------------------------
# PROBLEM DEFINITION
# --------------------------------------------------

def problem_definition(df: pd.DataFrame) -> dict:
    """
    Analyze a dataset and identify the likely ML problem,
    possible target column, and potential data-quality issues.
    """

    if df is None or df.empty:
        return {
            "success": False,
            "message": "Dataset is empty.",
        }

    try:
        # 1. Collect basic dataset information
        dataset_shape = df.shape
        columns = df.columns.tolist()
        data_types = df.dtypes.astype(str).to_dict()

        missing_values = df.isnull().sum().to_dict()

        missing_percentage = (
            (df.isnull().mean() * 100).round(2).to_dict()
        )

        duplicate_count = int(df.duplicated().sum())

        # 2. Generate a statistical summary
        summary = df.describe(include="all").to_string()
        sample_rows = df.head(5).to_string()

        # 3. Prepare information for the LLM
        dataset_info = f"""
Dataset shape: {dataset_shape}

Column names: {columns}

Data types: {data_types}

Missing values: {missing_values}

Missing percentage: {missing_percentage}

Duplicate rows: {duplicate_count}

Statistical summary:
{summary}

Sample rows:
{sample_rows}
"""

        prompt = f"""
You are a data analyst helping prepare a dataset for machine learning.

Analyze the following dataset information:

{dataset_info}

Return your analysis using these headings:

1. PROBLEM TYPE
   Identify whether this is likely classification, regression,
   clustering, or another ML task. Explain why.

2. TARGET COLUMN
   Suggest a possible target column if the evidence supports it.
   If the target cannot be determined, explicitly say so.

3. DATA QUALITY ISSUES
   Identify potential missing-value problems, duplicates,
   constant columns, ID columns, high-cardinality columns,
   possible data leakage, and class imbalance where applicable.

4. RECOMMENDED NEXT STEPS
   Suggest what should be inspected or addressed before
   training a model.

Rules:
- Do not modify the dataset.
- Do not assume the last column is the target.
- Distinguish confirmed observations from possibilities.
- Do not claim leakage or class imbalance without evidence.
- Keep the explanation clear and concise.
"""

        # 4. Load the Groq API key
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return {
                "success": False,
                "message": (
                    "GROQ_API_KEY is not configured. "
                    "Check that your .env file exists in the "
                    "project root and contains GROQ_API_KEY."
                ),
            }

        # 5. Initialize the LLM
        llm = ChatGroq(
     model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)

        # 6. Invoke the LLM
        response = llm.invoke(prompt)

        # 7. Return the analysis without modifying the DataFrame
        return {
            "success": True,
            "problem_summary": response.content,
            "dataset_shape": dataset_shape,
            "columns": columns,
            "data_types": data_types,
            "missing_values": missing_values,
            "missing_percentage": missing_percentage,
            "duplicate_count": duplicate_count,
            "target_column": None,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Problem definition failed: {str(e)}",
        }
