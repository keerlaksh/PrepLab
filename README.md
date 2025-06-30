# 🧠 LLM-Powered Data Preprocessing Pipeline with Streamlit + LangGraph

This project is a **LangGraph-based, LLM-assisted data preprocessing pipeline**, built using `Streamlit`, `LangGraph`, and `LangChain`. It automates the preprocessing of tabular data for machine learning tasks, including problem identification, data cleaning, missing value handling, outlier detection, and categorical encoding — all powered by LLaMA-3 through Groq.

---

## 🚀 Features

- 📂 Upload a CSV and let the app:
  - Define the ML problem (regression or classification)
  - Suggest and identify the target column
  - Inspect dataset structure and column types
  - Drop irrelevant/redundant columns using LLM analysis
  - Handle missing data with intelligent imputation
  - Remove outliers using Z-score or IQR (via LLM code)
  - Encode categorical variables with optimal methods (OneHot, Label, Frequency, Hashing, etc.)
- 🔄 Automatically chains the entire preprocessing flow using **LangGraph**
- 🤖 All preprocessing steps are guided by **LLM prompts**
- 📈 Generates boxplots before and after outlier removal
- 💾 Option to download generated preprocessing code
