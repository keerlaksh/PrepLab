import os
import ast
import pandas as pd
import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.feature_extraction import FeatureHasher
import re

# 🧠 Setup LLM
def get_llm():
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0
    )

# State Type
class DataState(TypedDict):
    df: pd.DataFrame
    problem_summary: str
    target_column: str
    inspection_summary: str
    df_cleaned: pd.DataFrame
    dropped_columns: list
    llm_response: str
    cleaned_df: pd.DataFrame
    boxplots: dict
    report: str
    code: str
    code_file: BytesIO
    # Enhanced state tracking for step-by-step results
    step_results: dict

# Helper function to create downloadable CSV
def create_csv_download(df, filename):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue(), filename

# ➀ Problem Definition Node
def define_problem(state: DataState) -> DataState:
    """Analyze CSV and define the ML problem"""
    df = state["df"]
    llm = get_llm()
    
    csv_description = df.describe(include='all').to_string()
    sample_rows = df.head().to_string()
    prompt = f"""
You are a machine learning expert. Analyze the following CSV file deeply and provide the following:

1. A brief project objective and desired outcomes
2. The appropriate machine learning task (e.g., regression, classification) and why
3. A suitable target column (if possible)
4. Key insights from the data (distributions, data types, anomalies)

Return your output in this Python dictionary format:
{{
    "summary": "your full summary text here",
    "target_column": "name_of_column"
}}

=== CSV DESCRIPTION ===
{csv_description}

=== SAMPLE ROWS ===
{sample_rows}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        result = ast.literal_eval(response.content.strip())
        summary = result["summary"]
        target_column = result["target_column"]
    except Exception:
        summary = response.content
        target_column = None
    
    state["problem_summary"] = summary
    state["target_column"] = target_column
    
    # Store step result
    if "step_results" not in state:
        state["step_results"] = {}
    state["step_results"]["problem_definition"] = {
        "df": df.copy(),
        "summary": summary,
        "target_column": target_column,
        "step_name": "Problem Definition"
    }
    
    return state

# ➁ Data Inspection Node
def inspect_data(state: DataState) -> DataState:
    """Inspect and explore the data"""
    df = state["df"]
    
    duplicated = df.duplicated().sum()
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    num_col = [col for col in df.columns if df[col].dtype != 'object']
    cat_unique = df[cat_col].nunique().to_dict()

    summary = f"""
🔍 **Data Inspection Summary**
```python
# Check duplicates and column types
duplicated = df.duplicated().sum()
cat_col = [col for col in df.columns if df[col].dtype == 'object']
num_col = [col for col in df.columns if df[col].dtype != 'object']
cat_unique = df[cat_col].nunique()
```
- Duplicate rows: {duplicated}
- Categorical columns: {cat_col}
- Numerical columns: {num_col}
- Unique values in categorical columns:
"""
    for col, count in cat_unique.items():
        summary += f"  - {col}: {count} unique values\n"
    
    state["inspection_summary"] = summary
    
    # Store step result
    state["step_results"]["data_inspection"] = {
        "df": df.copy(),
        "summary": summary,
        "step_name": "Data Inspection",
        "duplicated": duplicated,
        "categorical_cols": cat_col,
        "numerical_cols": num_col
    }
    
    return state

# ➂ Data Cleaning Node
def clean_columns(state: DataState) -> DataState:
    """Remove unwanted observations and columns"""
    df = state["df"]
    target_column = state["target_column"]
    problem_summary = state["problem_summary"]
    llm = get_llm()
    
    csv_description = df.describe(include='all').to_string()
    sample_rows = df.head().to_string()
    prompt = f"""
You are a data scientist working on a machine learning project.

Goal: Predict `{target_column}`.
Context: {problem_summary}

Duplicate observations most frequently arise during data collection and irrelevant observations are those that don't actually fit with the specific problem that we're trying to solve.

Redundant observations alter the efficiency and may produce misleading results.

Now, based on this goal and the structure of the dataset, identify:
- Columns to drop and reasons
- Columns to keep and reasons

Return Python dict like:
```python
{{
  "drop": {{
    "col1": "reason",
    "col2": "reason"
  }},
  "keep": {{
    "col3": "reason",
    "col4": "reason"
  }}
}}
=== CSV DESCRIPTION ===
{csv_description}

=== SAMPLE ROWS ===
{sample_rows}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        block = response.content[response.content.index("{"):response.content.rindex("}") + 1]
        result_dict = ast.literal_eval(block)
        drop_cols = list(result_dict.get("drop", {}).keys())
        drop_reasons = result_dict.get("drop", {})
        keep_reasons = result_dict.get("keep", {})
    except Exception:
        drop_cols, drop_reasons, keep_reasons = [], {}, {}

    df_cleaned = df.drop(columns=drop_cols, errors="ignore")
    explanation = f"""
👩‍💼 **LLM Data Cleaning Summary**
```python
# Drop columns suggested by LLM
df_cleaned = df.drop(columns={drop_cols}, errors='ignore')
```
"""
    if drop_cols:
        for col, reason in drop_reasons.items():
            explanation += f"- `{col}`: {reason}\n"
    else:
        explanation += f"✅ All columns are relevant for predicting `{target_column}`.\n"
    explanation += "\n🔎 **Columns kept:**\n"
    for col, reason in keep_reasons.items():
        explanation += f"- `{col}`: {reason}\n"
    
    state["df_cleaned"] = df_cleaned
    state["dropped_columns"] = drop_cols
    state["inspection_summary"] = state["inspection_summary"] + "\n\n" + explanation
    
    # Store step result
    state["step_results"]["column_cleaning"] = {
        "df": df_cleaned.copy(),
        "summary": explanation,
        "step_name": "Column Cleaning",
        "dropped_columns": drop_cols,
        "drop_reasons": drop_reasons,
        "keep_reasons": keep_reasons
    }
    
    return state

# ➃ Handle Missing Data Node
def handle_missing(state: DataState) -> DataState:
    """Handle missing values in the dataset"""
    df = state["df_cleaned"]
    total_rows = df.shape[0]
    null_percent = round((df.isnull().sum() / total_rows) * 100, 2)
    explanation = """
🧹 **Missing Data Handling Summary**
```python
# Calculate % of missing values
missing_percent = round((df.isnull().sum()/df.shape[0]) * 100, 2)
```
"""

    missing_report = null_percent[null_percent > 0]
    if not missing_report.empty:
        explanation += "\n📊 Columns with Missing Values:\n"
        for col, percent in missing_report.items():
            explanation += f"- `{col}`: {percent}% missing\n"
    else:
        explanation += "✅ No missing values found in dataset.\n"

    dropped = []
    for col, percent in null_percent.items():
        if percent > 50:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)
            explanation += f"\n🗑️ Dropped `{col}` due to {percent}% missing values."
        elif percent > 0 and df[col].dtype != 'object':
            skewness = df[col].skew()
            if abs(skewness) > 1:
                df[col] = df[col].fillna(df[col].median())
                explanation += f"""

🥒 Imputed `{col}` using **median** (skewed distribution: skew={round(skewness, 2)}).
```python
df['{col}'] = df['{col}'].fillna(df['{col}'].median())
```
"""
            else:
                df[col] = df[col].fillna(df[col].mean())
                explanation += f"""

📂 Imputed `{col}` using **mean** (normal-ish distribution: skew={round(skewness, 2)}).
```python
df['{col}'] = df['{col}'].fillna(df['{col}'].mean())
```
"""

    state["df_cleaned"] = df
    if "dropped_columns" not in state:
        state["dropped_columns"] = []
    state["dropped_columns"].extend(dropped)
    state["inspection_summary"] += "\n\n" + explanation
    
    # Store step result
    state["step_results"]["missing_data"] = {
        "df": df.copy(),
        "summary": explanation,
        "step_name": "Missing Data Handling",
        "missing_report": missing_report.to_dict(),
        "dropped_columns": dropped
    }
    
    return state

def extract_code_blocks(llm_response, columns):
    """Extract Python code blocks for each column from LLM response."""
    code_blocks = {}
    for col in columns:
        # Look for code blocks mentioning the column
        pattern = rf"```python[\s\S]*?df\['{col}'\][\s\S]*?```"
        matches = re.findall(pattern, llm_response)
        if matches:
            # Remove the triple backticks and 'python' if present
            code = matches[0].replace('```python', '').replace('```', '').strip()
            code_blocks[col] = code
        else:
            # Try to find any code block for the column
            pattern2 = rf"df\['{col}'\][\s\S]*?\n"
            matches2 = re.findall(pattern2, llm_response)
            if matches2:
                code_blocks[col] = matches2[0]
    return code_blocks

# ➄ Remove Outliers Node
def remove_outliers(state: DataState) -> DataState:
    """Remove outliers from numerical columns using LLM-suggested code if possible."""
    df = state["df_cleaned"]
    llm = get_llm()

    # Generate outlier analysis summary using LLM
    numeric_summary = df.describe().to_string()
    prompt = f"""
You are a machine learning expert. Analyze the following dataset statistics and decide the best outlier detection method (Z-score or IQR) for each column.

Return a detailed explanation of:
1. Why outliers matter.
2. For each column:
   - Whether it is normally distributed or skewed (decide using statistical hints)
   - Best method to detect outliers (Z-score or IQR)
   - Python code to visualize and clean them
   - Whether to trim or cap based on best practice
   - Why the method was chosen

Use this format for each column:
python
# Boxplot
sns.boxplot(df['COLUMN'])

# Z-score bounds
mean = df['COLUMN'].mean()
std = df['COLUMN'].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
filtered_df = df[(df['COLUMN'] >= lower_limit) & (df['COLUMN'] <= upper_limit)]

or
python
# IQR bounds
q1 = df['COLUMN'].quantile(0.25)
q3 = df['COLUMN'].quantile(0.75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
filtered_df = df[(df['COLUMN'] >= lower_limit) & (df['COLUMN'] <= upper_limit)]

Explain whether the outliers are to be removed (trimmed) or capped and why.

=== NUMERIC SUMMARY ===
{numeric_summary}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["llm_response"] = response.content

    cleaned_df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    boxplots = {}

    # Try to extract and execute LLM code for each column
    code_blocks = extract_code_blocks(response.content, numeric_cols)
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        sns.boxplot(x=df[col], ax=axes[0])
        axes[0].set_title(f"Before - {col}")
        local_vars = {'df': cleaned_df.copy(), 'sns': sns, 'plt': plt}
        code = code_blocks.get(col)
        try:
            if code:
                exec(code, globals(), local_vars)
                # LLM code should produce 'filtered_df'
                if 'filtered_df' in local_vars:
                    cleaned_df = local_vars['filtered_df']
            else:
                # Fallback to your logic
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                iqr_lower = q1 - 1.5 * iqr
                iqr_upper = q3 + 1.5 * iqr
                mean = df[col].mean()
                std = df[col].std()
                z_lower = mean - 3 * std
                z_upper = mean + 3 * std
                if abs(df[col].skew()) > 1:
                    lower_limit = iqr_lower
                    upper_limit = iqr_upper
                else:
                    lower_limit = z_lower
                    upper_limit = z_upper
                cleaned_df = cleaned_df[(cleaned_df[col] >= lower_limit) & (cleaned_df[col] <= upper_limit)]
        except Exception as e:
            # Fallback to your logic if LLM code fails
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            mean = df[col].mean()
            std = df[col].std()
            z_lower = mean - 3 * std
            z_upper = mean + 3 * std
            if abs(df[col].skew()) > 1:
                lower_limit = iqr_lower
                upper_limit = iqr_upper
            else:
                lower_limit = z_lower
                upper_limit = z_upper
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_limit) & (cleaned_df[col] <= upper_limit)]
        sns.boxplot(x=cleaned_df[col], ax=axes[1])
        axes[1].set_title(f"After - {col}")
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        boxplots[col] = buffer.read()
        plt.close(fig)

    state["cleaned_df"] = cleaned_df
    state["boxplots"] = boxplots
    state["step_results"]["outlier_removal"] = {
        "df": cleaned_df.copy(),
        "summary": response.content,
        "step_name": "Outlier Removal",
        "boxplots": boxplots,
        "numeric_cols": list(numeric_cols)
    }
    return state

# ➅ Encode Categoricals Node
def encode_categoricals(state: DataState) -> DataState:
    """Encode categorical variables using LLM-suggested code if possible."""
    df = state["cleaned_df"].copy()
    report = state.get("report", "")
    llm = get_llm()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        report += "\n✅ No categorical columns found.\n"
        state["df"] = df
        state["report"] = report
        state["code"] = ""
        state["code_file"] = None
        state["step_results"]["categorical_encoding"] = {
            "df": df.copy(),
            "summary": report,
            "step_name": "Categorical Encoding",
            "encoded_columns": []
        }
        return state
    sample = df[cat_cols].head(10).to_string()
    prompt = f"""
You are a machine learning data preprocessing expert.
Analyze each categorical column below and choose the best encoding method from:
Label Encoding, One-Hot Encoding, Ordinal Encoding, Frequency Encoding, Binary Encoding, Hashing Encoding, Target Encoding.
As we all know that better encoding leads to a better model and most algorithms cannot handle the categorical variables unless they are converted into a numerical value.

Categorical features are generally divided into 3 types: 

A. Binary: Either/or 
Examples: 

Yes, No
True, False
B. Ordinal: Specific ordered Groups. 
Examples: 

low, medium, high
cold, hot, lava Hot
C. Nominal: Unordered Groups. Examples 

cat, dog, tiger
pizza, burger, coke

Explain the type (Binary, Ordinal, Nominal), encoding method, and why it's chosen. Output the Python code for each transformation.
i want you analyze it deeply and  choose the correct encoding 

=== SAMPLE DATA ===
{sample}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    explanation = response.content
    code_lines = [
        "# Automatically generated encoding code\n",
        "import pandas as pd",
        "from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder",
        "from category_encoders import BinaryEncoder, TargetEncoder",
        "from sklearn.feature_extraction import FeatureHasher",
    ]
    encoded_columns = []
    code_blocks = extract_code_blocks(explanation, cat_cols)
    for col in cat_cols:
        code = code_blocks.get(col)
        try:
            if code:
                local_vars = {'df': df.copy(), 'LabelEncoder': LabelEncoder, 'OrdinalEncoder': OrdinalEncoder, 'OneHotEncoder': OneHotEncoder, 'BinaryEncoder': BinaryEncoder, 'TargetEncoder': TargetEncoder, 'FeatureHasher': FeatureHasher, 'pd': pd}
                exec(code, globals(), local_vars)
                if 'df' in local_vars:
                    df = local_vars['df']
                encoded_columns.append(f"{col} (LLM Encoded)")
                code_lines.append(code)
            else:
                # Fallback to your logic
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    report += f"\n✅ Label Encoded (Binary) '{col}'"
                    code_lines.append(f"df['{col}'] = LabelEncoder().fit_transform(df['{col}'])")
                    encoded_columns.append(f"{col} (Label Encoded)")
                elif unique_vals <= 10:
                    ohe = pd.get_dummies(df[col], prefix=col)
                    df.drop(columns=[col], inplace=True)
                    df = pd.concat([df, ohe], axis=1)
                    report += f"\n✅ One-Hot Encoded '{col}'"
                    code_lines.append(f"df = pd.concat([df, pd.get_dummies(df['{col}'], prefix='{col}')], axis=1); df.drop(columns=['{col}'], inplace=True)")
                    encoded_columns.append(f"{col} (One-Hot Encoded)")
                elif unique_vals <= 50:
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map)
                    report += f"\n✅ Frequency Encoded '{col}'"
                    code_lines.append(f"df['{col}'] = df['{col}'].map(df['{col}'].value_counts(normalize=True).to_dict())")
                    encoded_columns.append(f"{col} (Frequency Encoded)")
                else:
                    h = FeatureHasher(n_features=4, input_type='string')
                    hashed = h.fit_transform(df[col].astype(str)).toarray()
                    hashed_df = pd.DataFrame(hashed, columns=[f"{col}_hash_{i}" for i in range(hashed.shape[1])])
                    df.drop(columns=[col], inplace=True)
                    df = pd.concat([df, hashed_df], axis=1)
                    report += f"\n✅ Hash Encoded '{col}'"
                    code_lines.append(f"# Hash Encoding for '{col}'\nfrom sklearn.feature_extraction import FeatureHasher\nh = FeatureHasher(n_features=4, input_type='string')\nhashed = h.fit_transform(df['{col}'].astype(str)).toarray()\nhashed_df = pd.DataFrame(hashed, columns=[f'{col}_hash_' + str(i) for i in range(hashed.shape[1])])\ndf.drop(columns=['{col}'], inplace=True)\ndf = pd.concat([df, hashed_df], axis=1)")
                    encoded_columns.append(f"{col} (Hash Encoded)")
        except Exception as e:
            # Fallback to your logic if LLM code fails
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                report += f"\n✅ Label Encoded (Binary) '{col}'"
                code_lines.append(f"df['{col}'] = LabelEncoder().fit_transform(df['{col}'])")
                encoded_columns.append(f"{col} (Label Encoded)")
            elif unique_vals <= 10:
                ohe = pd.get_dummies(df[col], prefix=col)
                df.drop(columns=[col], inplace=True)
                df = pd.concat([df, ohe], axis=1)
                report += f"\n✅ One-Hot Encoded '{col}'"
                code_lines.append(f"df = pd.concat([df, pd.get_dummies(df['{col}'], prefix='{col}')], axis=1); df.drop(columns=['{col}'], inplace=True)")
                encoded_columns.append(f"{col} (One-Hot Encoded)")
            elif unique_vals <= 50:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map)
                report += f"\n✅ Frequency Encoded '{col}'"
                code_lines.append(f"df['{col}'] = df['{col}'].map(df['{col}'].value_counts(normalize=True).to_dict())")
                encoded_columns.append(f"{col} (Frequency Encoded)")
            else:
                h = FeatureHasher(n_features=4, input_type='string')
                hashed = h.fit_transform(df[col].astype(str)).toarray()
                hashed_df = pd.DataFrame(hashed, columns=[f"{col}_hash_{i}" for i in range(hashed.shape[1])])
                df.drop(columns=[col], inplace=True)
                df = pd.concat([df, hashed_df], axis=1)
                report += f"\n✅ Hash Encoded '{col}'"
                code_lines.append(f"# Hash Encoding for '{col}'\nfrom sklearn.feature_extraction import FeatureHasher\nh = FeatureHasher(n_features=4, input_type='string')\nhashed = h.fit_transform(df['{col}'].astype(str)).toarray()\nhashed_df = pd.DataFrame(hashed, columns=[f'{col}_hash_' + str(i) for i in range(hashed.shape[1])])\ndf.drop(columns=['{col}'], inplace=True)\ndf = pd.concat([df, hashed_df], axis=1)")
                encoded_columns.append(f"{col} (Hash Encoded)")
    code_str = "\n".join(code_lines)
    buffer = BytesIO(code_str.encode())
    buffer.name = "encoding_code.py"
    state["df"] = df
    state["report"] = report + "\n\n🧠 LLM Analysis:\n" + explanation
    state["code"] = code_str
    state["code_file"] = buffer
    state["step_results"]["categorical_encoding"] = {
        "df": df.copy(),
        "summary": report + "\n\n🧠 LLM Analysis:\n" + explanation,
        "step_name": "Categorical Encoding",
        "encoded_columns": encoded_columns,
        "code": code_str
    }
    return state

# Build the LangGraph
def create_preprocessing_graph():
    """Create the preprocessing graph with specified flow"""
    builder = StateGraph(DataState)
    
    # Add nodes
    builder.add_node("define_problem", define_problem)
    builder.add_node("inspect_data", inspect_data)
    builder.add_node("clean_columns", clean_columns)
    builder.add_node("handle_missing", handle_missing)
    builder.add_node("remove_outliers", remove_outliers)
    builder.add_node("encode_categoricals", encode_categoricals)
    
    # Set entry point and edges
    builder.set_entry_point("define_problem")
    builder.add_edge("define_problem", "inspect_data")
    builder.add_edge("inspect_data", "clean_columns")
    builder.add_edge("clean_columns", "handle_missing")
    builder.add_edge("handle_missing", "remove_outliers")
    builder.add_edge("remove_outliers", "encode_categoricals")
    builder.add_edge("encode_categoricals", END)
    
    return builder.compile()

# Function to display step results with enhanced UI
def display_step_results(step_key, step_data, step_number):
    """Display results for each preprocessing step with enhanced UI"""
    with st.container():
        # Create a card-like appearance
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="color: white; margin: 0;">
                Step {step_number}: {step_data['step_name']}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display summary
            if 'summary' in step_data:
                with st.expander(f"📋 {step_data['step_name']} Details", expanded=False):
                    st.markdown(step_data['summary'])
            
            # Display dataframe preview
            with st.expander(f"👀 Data Preview ({step_data['df'].shape[0]} rows × {step_data['df'].shape[1]} cols)", expanded=False):
                st.dataframe(step_data['df'].head(10))
                
                # Show data info
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Rows", step_data['df'].shape[0])
                with info_col2:
                    st.metric("Columns", step_data['df'].shape[1])
                with info_col3:
                    st.metric("Memory Usage", f"{step_data['df'].memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        with col2:
            # Download button
            csv_data, filename = create_csv_download(step_data['df'], f"step_{step_number}_{step_key}.csv")
            st.download_button(
                label=f"📥 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key=f"download_{step_key}_{step_number}",
                help=f"Download data after {step_data['step_name']}"
            )
            
            # Display specific metrics for each step
            if step_key == "data_inspection":
                if 'duplicated' in step_data:
                    st.metric("Duplicates", step_data['duplicated'])
                if 'categorical_cols' in step_data:
                    st.metric("Cat. Columns", len(step_data['categorical_cols']))
                if 'numerical_cols' in step_data:
                    st.metric("Num. Columns", len(step_data['numerical_cols']))
            
            elif step_key == "column_cleaning":
                if 'dropped_columns' in step_data:
                    st.metric("Dropped Columns", len(step_data['dropped_columns']))
            
            elif step_key == "missing_data":
                if 'missing_report' in step_data:
                    st.metric("Cols with Missing", len(step_data['missing_report']))
            
            elif step_key == "categorical_encoding":
                if 'encoded_columns' in step_data:
                    st.metric("Encoded Columns", len(step_data['encoded_columns']))

# Streamlit Interface
def main():
    st.set_page_config(
        page_title="🤖 AI-Powered Data Preprocessing Pipeline",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Data Preprocessing Pipeline</h1>
        <p style="font-size: 1.2em; margin: 0;">Automated data preprocessing using LangGraph and LLM intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API key and pipeline info
    with st.sidebar:
        st.markdown("### 🔑 Configuration")
        groq_api_key = st.text_input("GROQ API Key", type="password", help="Enter your GROQ API key")
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
            st.success("✅ API Key configured!")
        
        st.markdown("---")
        st.markdown("### 🔄 Pipeline Steps")
        st.markdown("""
        <div style="background: black; padding: 1rem; border-radius: 8px;">
        1. 🎯 <b>Define Problem</b><br>
        &nbsp;&nbsp;&nbsp;AI analyzes dataset and identifies ML task<br><br>
        2. 🔍 <b>Inspect Data</b><br>
        &nbsp;&nbsp;&nbsp;Explore data structure and types<br><br>
        3. 🧹 <b>Clean Columns</b><br>
        &nbsp;&nbsp;&nbsp;Remove irrelevant columns<br><br>
        4. 🩹 <b>Handle Missing</b><br>
        &nbsp;&nbsp;&nbsp;Impute or remove missing values<br><br>
        5. 📊 <b>Remove Outliers</b><br>
        &nbsp;&nbsp;&nbsp;Detect and handle outliers<br><br>
        6. 🔢 <b>Encode Categoricals</b><br>
        &nbsp;&nbsp;&nbsp;Transform categorical variables
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("""
        - ✨ **Step-by-step preview**
        - 📥 **Download after each step**
        - 🤖 **AI-powered decisions**
        - 📈 **Visual outlier analysis**
        - 💾 **Generated code download**
        """)
    
    # File upload section
    st.markdown("### 📁 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your CSV file to start the preprocessing pipeline"
    )
    
    if uploaded_file is not None and groq_api_key:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display upload success with metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Rows", df.shape[0])
            with col2:
                st.metric("📋 Columns", df.shape[1])
            with col3:
                st.metric("💾 Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("🔍 Missing Values", df.isnull().sum().sum())
            
            # Show original data preview
            with st.expander("👀 Preview Original Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Types:**")
                    st.write(df.dtypes.value_counts())
                with col2:
                    st.write("**Columns:**")
                    st.write(list(df.columns))
            
            # Run preprocessing pipeline
            st.markdown("### 🚀 Start Processing")
            if st.button("🚀 Run Preprocessing Pipeline", type="primary", use_container_width=True):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("🔄 Running AI-powered preprocessing pipeline..."):
                    # Initialize state
                    initial_state = DataState(
                        df=df,
                        problem_summary="",
                        target_column="",
                        inspection_summary="",
                        df_cleaned=pd.DataFrame(),
                        dropped_columns=[],
                        llm_response="",
                        cleaned_df=pd.DataFrame(),
                        boxplots={},
                        report="",
                        code="",
                        code_file=None,
                        step_results={}
                    )
                    
                    # Create and run graph
                    graph = create_preprocessing_graph()
                    
                    # Update progress as we go through steps
                    steps = ["define_problem", "inspect_data", "clean_columns", "handle_missing", "remove_outliers", "encode_categoricals"]
                    
                    for i, step in enumerate(steps):
                        progress_bar.progress((i + 1) / len(steps))
                        status_text.text(f"Running step {i + 1}/6: {step.replace('_', ' ').title()}")
                        
                    result = graph.invoke(initial_state)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Pipeline completed successfully!")
                    
                    # Display results
                    st.success("🎉 Preprocessing pipeline completed successfully!")
                    
                    # Display step-by-step results
                    st.markdown("### 📊 Step-by-Step Results")
                    
                    if "step_results" in result:
                        step_order = [
                            ("problem_definition", "Problem Definition"),
                            ("data_inspection", "Data Inspection"), 
                            ("column_cleaning", "Column Cleaning"),
                            ("missing_data", "Missing Data Handling"),
                            ("outlier_removal", "Outlier Removal"),
                            ("categorical_encoding", "Categorical Encoding")
                        ]
                        
                        # Create tabs for better organization
                        tab_names = [f"{i+1}. {name}" for i, (_, name) in enumerate(step_order)]
                        tabs = st.tabs(tab_names)
                        
                        for i, (step_key, step_name) in enumerate(step_order):
                            if step_key in result["step_results"]:
                                with tabs[i]:
                                    display_step_results(step_key, result["step_results"][step_key], i+1)
                                    
                                    # Special handling for outlier removal step
                                    if step_key == "outlier_removal" and "boxplots" in result["step_results"][step_key]:
                                        st.markdown("#### 📈 Outlier Visualization")
                                        boxplots = result["step_results"][step_key]["boxplots"]
                                        if boxplots:
                                            cols = st.columns(min(2, len(boxplots)))
                                            for idx, (col_name, plot_data) in enumerate(boxplots.items()):
                                                with cols[idx % 2]:
                                                    st.image(plot_data, caption=f"Before/After: {col_name}")
                    
                    # Final summary section
                    st.markdown("### 🎯 Processing Summary")
                    
                    final_df = result.get("df", result.get("cleaned_df", df))
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Original Rows", df.shape[0])
                    with col2:
                        st.metric("Final Rows", final_df.shape[0], delta=final_df.shape[0] - df.shape[0])
                    with col3:
                        st.metric("Original Columns", df.shape[1])
                    with col4:
                        st.metric("Final Columns", final_df.shape[1], delta=final_df.shape[1] - df.shape[1])
                    
                    # Problem definition results
                    if result.get("problem_summary"):
                        with st.expander("🎯 AI Problem Analysis", expanded=True):
                            st.markdown(result["problem_summary"])
                            if result.get("target_column"):
                                st.info(f"🎯 **Suggested Target Column:** `{result['target_column']}`")
                    
                    # Final processed dataset
                    st.markdown("### 🎉 Final Processed Dataset")
                    with st.expander("👀 Preview Final Dataset", expanded=True):
                        st.dataframe(final_df.head(10), use_container_width=True)
                        
                        # Data quality metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Missing Values", final_df.isnull().sum().sum())
                        with col2:
                            st.metric("Duplicate Rows", final_df.duplicated().sum())
                        with col3:
                            numeric_cols = final_df.select_dtypes(include=['int64', 'float64']).columns
                            st.metric("Numeric Columns", len(numeric_cols))
                    
                    # Download section
                    st.markdown("### 📥 Download Results")
                    
                    download_col1, download_col2, download_col3 = st.columns(3)
                    
                    with download_col1:
                        # Download processed CSV
                        csv_data, _ = create_csv_download(final_df, "final_processed_data.csv")
                        st.download_button(
                            label="📊 Download Final CSV",
                            data=csv_data,
                            file_name="final_processed_data.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="Download the final processed dataset"
                        )
                    
                    with download_col2:
                        # Download processing code
                        if result.get("code"):
                            st.download_button(
                                label="💻 Download Code",
                                data=result["code"],
                                file_name="preprocessing_code.py",
                                mime="text/python",
                                use_container_width=True,
                                help="Download the Python code for preprocessing"
                            )
                    
                    with download_col3:
                        # Download processing report
                        if result.get("report"):
                            st.download_button(
                                label="📋 Download Report",
                                data=result["report"],
                                file_name="preprocessing_report.md",
                                mime="text/markdown",
                                use_container_width=True,
                                help="Download the preprocessing report"
                            )
                    
                    # Additional insights
                    st.markdown("### 💡 Key Insights")
                    insights_col1, insights_col2 = st.columns(2)
                    
                    with insights_col1:
                        st.markdown("#### 🔄 Data Transformations")
                        transformations = []
                        if result.get("dropped_columns"):
                            transformations.append(f"• Dropped {len(result['dropped_columns'])} columns")
                        
                        # Check for missing data handling
                        if "step_results" in result and "missing_data" in result["step_results"]:
                            missing_info = result["step_results"]["missing_data"]
                            if missing_info.get("missing_report"):
                                transformations.append(f"• Handled missing data in {len(missing_info['missing_report'])} columns")
                        
                        # Check for outlier removal
                        if "step_results" in result and "outlier_removal" in result["step_results"]:
                            outlier_info = result["step_results"]["outlier_removal"]
                            if outlier_info.get("numeric_cols"):
                                transformations.append(f"• Removed outliers from {len(outlier_info['numeric_cols'])} numeric columns")
                        
                        # Check for encoding
                        if "step_results" in result and "categorical_encoding" in result["step_results"]:
                            encoding_info = result["step_results"]["categorical_encoding"]
                            if encoding_info.get("encoded_columns"):
                                transformations.append(f"• Encoded {len(encoding_info['encoded_columns'])} categorical columns")
                        
                        for transformation in transformations:
                            st.markdown(transformation)
                    
                    with insights_col2:
                        st.markdown("#### 📈 Data Quality Improvements")
                        improvements = []
                        
                        # Calculate improvements
                        original_missing = df.isnull().sum().sum()
                        final_missing = final_df.isnull().sum().sum()
                        if original_missing > final_missing:
                            improvements.append(f"• Reduced missing values by {original_missing - final_missing}")
                        
                        original_duplicates = df.duplicated().sum()
                        final_duplicates = final_df.duplicated().sum()
                        if original_duplicates > final_duplicates:
                            improvements.append(f"• Removed {original_duplicates - final_duplicates} duplicate rows")
                        
                        # Check for data type improvements
                        original_objects = len(df.select_dtypes(include=['object']).columns)
                        final_objects = len(final_df.select_dtypes(include=['object']).columns)
                        if original_objects > final_objects:
                            improvements.append(f"• Converted {original_objects - final_objects} categorical columns to numeric")
                        
                        if not improvements:
                            improvements.append("• Dataset was already in good condition")
                        
                        for improvement in improvements:
                            st.markdown(improvement)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.markdown("**Troubleshooting Tips:**")
            st.markdown("• Ensure your CSV file is properly formatted")
            st.markdown("• Check that your GROQ API key is correct")
            st.markdown("• Try with a smaller dataset first")
    
    elif uploaded_file is not None and not groq_api_key:
        st.warning("⚠️ Please enter your GROQ API key in the sidebar to proceed.")
    
    elif not uploaded_file:
        # Welcome section with instructions
        st.markdown("### 🚀 Getting Started")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### Welcome to the AI-Powered Data Preprocessing Pipeline!
            
            This intelligent tool will automatically analyze and preprocess your dataset using advanced AI techniques. Here's what it does:
            
            **🤖 AI-Driven Analysis:**
            - Automatically identifies the ML problem type
            - Suggests appropriate target columns
            - Makes intelligent preprocessing decisions
            
            **📊 Comprehensive Preprocessing:**
            - Data inspection and quality assessment
            - Smart column selection and cleaning
            - Intelligent missing value handling
            - Automated outlier detection and removal
            - Optimal categorical encoding selection
            
            **✨ Enhanced Features:**
            - Preview data after each step
            - Download CSV files at any stage
            - Visual outlier analysis with before/after plots
            - Generated Python code for reproducibility
            - Detailed processing reports
            """)
        
        with col2:
            st.markdown("#### 📋 Requirements")
            st.info("""
            **What you need:**
            - A CSV file with your data
            - GROQ API key (free at groq.com)
            - Internet connection
            
            **Supported formats:**
            - CSV files only
            - Any size (larger files take longer)
            - Mixed data types supported
            """)
            
            st.markdown("#### 🎯 Best Practices")
            st.success("""
            **For best results:**
            - Ensure column names are descriptive
            - Include a clear target variable
            - Remove any sensitive information
            - Use standard CSV format
            """)

if __name__ == "__main__":
    main()