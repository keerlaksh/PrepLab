
# tools/clean_columns.py

import os
import json
import re
import pandas as pd
from langchain_groq import ChatGroq


def clean_columns(
    df: pd.DataFrame,
    target_column: str = None,
    problem_summary: str = None,
    missing_threshold: float = 0.50,
    high_cardinality_threshold: float = 0.90,
    auto_drop_high_missing: bool = True,
    use_llm: bool = True
) -> dict:
    """
    Select features using deterministic checks and optional LLM analysis.

    Automatically handles constant columns and, optionally, columns
    exceeding the missingness threshold.

    ID-like, high-cardinality, irrelevant, and possible leakage columns
    are assessed separately. The target column is protected.
    """

    if df is None or df.empty:
        return {"success": False, "message": "Dataset is empty."}

    if not 0 <= missing_threshold <= 1:
        return {
            "success": False,
            "message": "missing_threshold must be between 0 and 1."
        }

    if not 0 <= high_cardinality_threshold <= 1:
        return {
            "success": False,
            "message": "high_cardinality_threshold must be between 0 and 1."
        }

    if target_column is not None and target_column not in df.columns:
        return {
            "success": False,
            "message": f"Target column '{target_column}' does not exist."
        }

    original_columns = df.columns.tolist()
    column_info = {}
    automatic_drops = {}
    review_flags = {}

    # 1. Perform objective feature-quality checks.
    for col in df.columns:
        series = df[col]
        missing_ratio = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))
        non_missing_count = int(series.notna().sum())

        unique_ratio = (
            unique_count / non_missing_count
            if non_missing_count > 0 else 0.0
        )

        name = str(col).lower()
        id_pattern = bool(
            re.search(
                r"(^id$|_id$|^id_|uuid|identifier|record_number)",
                name
            )
        )

        constant = non_missing_count > 0 and unique_count <= 1
        high_missing = missing_ratio > missing_threshold
        high_cardinality = (
            unique_count > 10
            and unique_ratio >= high_cardinality_threshold
        )

        column_info[col] = {
            "dtype": str(series.dtype),
            "missing_percentage": round(missing_ratio * 100, 2),
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 3),
            "constant": constant,
            "high_missingness": high_missing,
            "id_like_name": id_pattern,
            "high_cardinality": high_cardinality,
            "sample_values": (
                series.dropna().astype(str).head(3).tolist()
            )
        }

        if col == target_column:
            continue

        # Constant columns have no variation to help a model.
        if constant:
            automatic_drops[col] = (
                "Constant column: only one non-missing unique value."
            )

        # Configurable missingness rule.
        elif high_missing and auto_drop_high_missing:
            automatic_drops[col] = (
                f"{missing_ratio:.1%} missing values; "
                f"exceeds threshold of {missing_threshold:.0%}."
            )

        # ID and high-cardinality columns are flags, not automatic drops.
        if id_pattern:
            review_flags[col] = [
                "Column name resembles an identifier; "
                "check whether it contains predictive information."
            ]

        if high_cardinality:
            review_flags.setdefault(col, []).append(
                "Very high proportion of unique values; "
                "check whether the feature is useful and how it should "
                "be represented."
            )

        if high_missing and not auto_drop_high_missing:
            review_flags.setdefault(col, []).append(
                f"{missing_ratio:.1%} missing values; "
                "consider imputation or removal."
            )

    # 2. Ask the LLM for contextual decisions.
    # It is not responsible for repeating the deterministic checks.
    llm_drops = {}
    llm_review = {}
    llm_warning = None

    if use_llm:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            llm_warning = (
                "GROQ_API_KEY is missing; contextual LLM analysis "
                "was skipped."
            )
        else:
            try:
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    api_key=api_key
                )

                # Exclude features already selected for automatic removal.
                candidates = {
                    str(col): info
                    for col, info in column_info.items()
                    if col not in automatic_drops
                }

                prompt = f"""
You are performing contextual feature selection.

Prediction objective:
{problem_summary or "Not provided. Be conservative."}

Target column:
{target_column or "Unknown. Do not guess."}

Candidate column information:
{json.dumps(candidates, default=str)}

Choose only columns with a defensible contextual reason for removal.

Focus on:
- Possible target leakage: a feature that directly reveals the target
  or would not be available at prediction time.
- Irrelevant features: a feature with no plausible relationship to
  the stated prediction objective.

ID-like and high-cardinality columns are already flagged by Python.
Do not recommend dropping them merely because of their names or
unique-value counts.

Do not repeat automatic constant/missingness decisions.
Do not recommend dropping the target.
Do not invent columns.
When evidence is insufficient, flag the column for review instead.
Leakage cannot always be confirmed from column names and sample values.

Return only valid JSON:
{{
  "drop_columns": {{
    "column_name": "specific reason"
  }},
  "review_columns": {{
    "column_name": "concern and what needs verification"
  }}
}}
"""

                response = llm.invoke(prompt)
                content = response.content.strip()

                if content.startswith("```"):
                    content = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        content,
                        flags=re.IGNORECASE
                    ).strip()

                result = json.loads(content)

                proposed_drops = result.get("drop_columns", {})
                proposed_reviews = result.get("review_columns", {})

                if not isinstance(proposed_drops, dict):
                    raise ValueError("drop_columns must be a JSON object.")

                if not isinstance(proposed_reviews, dict):
                    raise ValueError("review_columns must be a JSON object.")

                for col, reason in proposed_drops.items():
                    if (
                        col in df.columns
                        and col != target_column
                        and col not in automatic_drops
                    ):
                        llm_drops[col] = str(reason)

                for col, reason in proposed_reviews.items():
                    if col in df.columns and col != target_column:
                        llm_review.setdefault(col, []).append(str(reason))

            except Exception as e:
                llm_warning = (
                    f"LLM analysis failed; deterministic checks "
                    f"were still performed. Details: {e}"
                )

    # 3. Combine findings and remove selected columns.
    all_drops = {**automatic_drops, **llm_drops}

    cleaned_df = df.drop(
        columns=list(all_drops.keys()),
        errors="ignore"
    ).copy()

    # 4. Produce an auditable report.
    removal_report = {
        col: {
            "reason": reason,
            "source": (
                "Python rule"
                if col in automatic_drops
                else "LLM recommendation"
            )
        }
        for col, reason in all_drops.items()
    }

    return {
        "success": True,
        "original_shape": df.shape,
        "cleaned_shape": cleaned_df.shape,
        "cleaned_df": cleaned_df,
        "dropped_columns": list(all_drops.keys()),
        "removal_report": removal_report,
        "review_flags": review_flags,
        "llm_review": llm_review,
        "column_evidence": column_info,
        "retained_columns": cleaned_df.columns.tolist(),
        "target_column": target_column,
        "llm_warning": llm_warning
    }