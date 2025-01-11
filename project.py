import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, wilcoxon
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt

# Pandas settings
pd.options.display.float_format = '{:,.4f}'.format

# Title
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Application</h1>", unsafe_allow_html=True)

# Functions

def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error while loading data: {e}")
        return None

def describe_columns(data):
    column_details = pd.DataFrame({
        "Column Name": data.columns,
        "Data Type": data.dtypes,
        "Unique Values": [data[col].nunique() for col in data.columns],
        "Missing Values": [data[col].isnull().sum() for col in data.columns]
    })
    return column_details

def run_shapiro_test(data):
    stat, p = shapiro(data)
    return stat, p

def run_levene_test(group1, group2):
    stat, p = levene(group1, group2)
    return stat, p

def display_test_results(test_name, stat, p_value):
    st.markdown(f"### {test_name} Results")
    st.write(f"- Test Statistic: {stat:.4f}")
    st.write(f"- P-value: {p_value:.4f}")
    if p_value > 0.05:
        st.success("Result: The null hypothesis cannot be rejected.")
    else:
        st.error("Result: The null hypothesis is rejected.")

# Step 1: Load Data
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 1: Load Dataset</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)

    if data is not None:
        # General Information
        st.write(f"**Number of Rows:** {data.shape[0]}")
        st.write(f"**Number of Columns:** {data.shape[1]}")

        # Column Details
        st.write("### Column Details")
        column_details = describe_columns(data)
        st.dataframe(column_details)

        # Data Preview
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Step 2: Select Columns and Define Groups
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Columns and Define Groups</h2>", unsafe_allow_html=True)
        selected_columns = st.multiselect("Select columns for hypothesis testing", data.columns.tolist())

        if not selected_columns:
            st.warning("Please select at least one column to proceed.")
            st.stop()

        for column in selected_columns:
            cleaned_data = data[column].dropna()

            if cleaned_data.empty:
                st.error(f"'{column}' column does not contain valid data for hypothesis testing.")
                continue

            if pd.api.types.is_numeric_dtype(cleaned_data):
                st.markdown(f"<h3 style='text-align: center;'>{column} Column Analysis (Numerical)</h3>", unsafe_allow_html=True)

                # Normality Test
                if st.checkbox(f"Perform Normality Test (Shapiro-Wilk) for {column}"):
                    stat, p_value = run_shapiro_test(cleaned_data)
                    display_test_results("Shapiro-Wilk Test", stat, p_value)

                    # Histogram
                    fig, ax = plt.subplots()
                    ax.hist(cleaned_data, bins=10, color='blue', edgecolor='black')
                    st.pyplot(fig)

                # Variance Homogeneity Test (Levene)
                if len(cleaned_data) >= 2 and st.checkbox(f"Perform Variance Homogeneity Test (Levene) for {column}"):
                    group1 = cleaned_data[:len(cleaned_data)//2]
                    group2 = cleaned_data[len(cleaned_data)//2:]
                    stat, p_value = run_levene_test(group1, group2)
                    display_test_results("Levene Test", stat, p_value)

            elif pd.api.types.is_categorical_dtype(cleaned_data) or isinstance(cleaned_data.iloc[0], str):
                st.markdown(f"<h3 style='text-align: center;'>{column} Column Analysis (Categorical)</h3>", unsafe_allow_html=True)
                st.warning("Categorical data analysis is currently limited.")

            else:
                st.error(f"'{column}' column contains unsupported data type. Please select columns with numerical or categorical data.")
    else:
        st.warning("The uploaded dataset could not be processed.")
else:
    st.info("Please upload a CSV file.")
