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

to run:
streamlit run app.py

![image](https://github.com/user-attachments/assets/a3a79809-6583-4251-9a39-3e0256d95690)
![image](https://github.com/user-attachments/assets/46a8af3e-cb0b-424b-a192-08c74a3f0bb2)
![image](https://github.com/user-attachments/assets/aae78a56-6757-4141-adcf-70d2696bdcee)
![image](https://github.com/user-attachments/assets/1768447d-29ae-41f0-adfe-fa662bbee8ce)
![image](https://github.com/user-attachments/assets/c9860995-96fa-4b02-aeb4-cd2131861789)
![image](https://github.com/user-attachments/assets/08178600-d16b-47af-8b0a-262b99293b3e)
![image](https://github.com/user-attachments/assets/29433e1c-899f-4545-ac2f-82c36fdb9449)
![image](https://github.com/user-attachments/assets/69cf9f53-c3c2-4603-bc3b-ac8ee4aae33f)
![image](https://github.com/user-attachments/assets/a9faa783-c650-4f8e-b328-0cf925529daf)
![image](https://github.com/user-attachments/assets/be468829-3cfb-4e48-a3e1-78cfda9ae5da)













