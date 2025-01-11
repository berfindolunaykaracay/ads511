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

def check_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers), outliers

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

                # Step 3: Define Hypothesis
                st.subheader(f"Define Hypotheses for {column}")
                st.write("**Null Hypothesis (H0):** No significant difference.")
                st.write("**Alternative Hypothesis (H1):** There is a significant difference.")

                # Step 4: Assumption Check
                st.subheader(f"Assumption Check for {column}")

                # Normality Test
                if st.checkbox(f"Perform Normality Test (Shapiro-Wilk) for {column}"):
                    stat, p_value = run_shapiro_test(cleaned_data)
                    display_test_results("Shapiro-Wilk Test", stat, p_value)

                # Homogeneity of Variance
                if len(cleaned_data) >= 2 and st.checkbox(f"Perform Variance Homogeneity Test (Levene) for {column}"):
                    group1 = cleaned_data[:len(cleaned_data)//2]
                    group2 = cleaned_data[len(cleaned_data)//2:]
                    stat, p_value = run_levene_test(group1, group2)
                    display_test_results("Levene Test", stat, p_value)

                # Outlier Check
                if st.checkbox(f"Check for Outliers in {column}"):
                    outlier_count, outliers = check_outliers(cleaned_data)
                    st.write(f"Number of Outliers: {outlier_count}")
                    if outlier_count > 0:
                        st.write("Outliers:")
                        st.write(outliers)

                # Independent and Identically Distributed (IID) Check
                st.write("**Note:** IID checks need external validation based on domain knowledge.")

                # Step 5: Decide Test Type
                st.subheader(f"Test Type Decision for {column}")
                parametric_conditions = st.checkbox("All Assumptions for Parametric Tests are Met")

                if parametric_conditions:
                    test_type = st.radio("Select a Parametric Test", ["Independent T-Test", "Paired T-Test"])
                else:
                    test_type = st.radio("Select a Non-Parametric Test", ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"])

                # Step 6: Evaluation
                if st.button(f"Run {test_type} for {column}"):
                    if test_type == "Independent T-Test":
                        stat, p_value = ttest_ind(group1, group2)
                        display_test_results("Independent T-Test", stat, p_value)
                    elif test_type == "Paired T-Test":
                        st.error("Paired T-Test requires paired data.")
                    elif test_type == "Mann-Whitney U Test":
                        stat, p_value = mannwhitneyu(group1, group2)
                        display_test_results("Mann-Whitney U Test", stat, p_value)
                    elif test_type == "Wilcoxon Signed-Rank Test":
                        stat, p_value = wilcoxon(group1, group2)
                        display_test_results("Wilcoxon Signed-Rank Test", stat, p_value)

            elif pd.api.types.is_categorical_dtype(cleaned_data) or isinstance(cleaned_data.iloc[0], str):
                st.markdown(f"<h3 style='text-align: center;'>{column} Column Analysis (Categorical)</h3>", unsafe_allow_html=True)
                st.subheader(f"Non-Parametric Analysis for {column}")
                st.write("Categorical data automatically uses non-parametric tests.")

                test_type = st.radio("Select a Test", ["Chi-Square Test", "Fisher's Exact Test"])

                if st.button(f"Run {test_type} for {column}"):
                    if test_type == "Chi-Square Test":
                        st.error("Chi-Square Test requires contingency tables.")
                    elif test_type == "Fisher's Exact Test":
                        st.error("Fisher's Exact Test requires contingency tables.")

            else:
                st.error(f"'{column}' column contains unsupported data type. Please select columns with numerical or categorical data.")
    else:
        st.warning("The uploaded dataset could not be processed.")
else:
    st.info("Please upload a CSV file.")
