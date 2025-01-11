import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency, fisher_exact
pd.options.display.float_format = '{:,.4f}'.format

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Application</h1>", unsafe_allow_html=True)

# Step 1: Veri Yükleme
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 1: Upload Dataset</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center;'>Dataset Preview</h2>", unsafe_allow_html=True)

    # Genel Bilgiler
    st.write(f"**Number of Rows:** {data.shape[0]}")
    st.write(f"**Number of Columns:** {data.shape[1]}")

    # Sütun Bilgileri
    st.write("### Column Information")
    column_details = pd.DataFrame({
        "Column Name": data.columns,
        "Data Type": data.dtypes,
        "Number of Unique Values": [data[col].nunique() for col in data.columns],
        "Number of Missing Values": [data[col].isnull().sum() for col in data.columns]
    })
    st.dataframe(column_details)

    # Veri Önizlemesi
    st.write("### Dataset Head")
    st.dataframe(data.head())

    # Step 2: Sütun Seçimi ve Grup Bilgisi
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Columns and Define Groups</h2>", unsafe_allow_html=True)
    selected_columns = st.multiselect("Select columns for hypothesis testing", data.columns.tolist())

    if not selected_columns:
        st.warning("Please select at least one column to proceed.")
        st.stop()

    column_types = {}
    all_groups = {}

    # Sütunların veri türlerini belirleme
    for selected_test_col in selected_columns:
        cleaned_data = data[selected_test_col].dropna()
        if cleaned_data.empty:
            st.error(f"Column '{selected_test_col}' has no valid data for hypothesis testing.")
            continue

        if pd.api.types.is_numeric_dtype(cleaned_data):
            column_types[selected_test_col] = "Numerical"
            all_groups[selected_test_col] = cleaned_data.tolist()
        elif pd.api.types.is_categorical_dtype(cleaned_data) or isinstance(cleaned_data.iloc[0], str):
            column_types[selected_test_col] = "Categorical"
            all_groups[selected_test_col] = cleaned_data.tolist()
        else:
            st.error(f"Column '{selected_test_col}' contains unsupported data type. Please select columns with either numeric or categorical data.")

    if not all_groups:
        st.warning("No valid data provided.")
        st.stop()

    # Seçilen sütunların veri türlerini gösterme
    st.write("### Selected Columns and Detected Data Types")
    detected_types = pd.DataFrame({"Column Name": column_types.keys(), "Data Type": column_types.values()})
    st.dataframe(detected_types)

    # Her tür için işlem önerileri
    for column, col_type in column_types.items():
        st.markdown(f"<h3 style='text-align: center;'>Analysis for Column: {column} ({col_type})</h3>", unsafe_allow_html=True)

        if col_type == "Numerical":
            st.write("### Recommended Tests for Numerical Data")
            methods = [
                "t_test_independent", "dependent_ttest", "repeated_measure_anova", "oneway_anova", "Wilcoxon_signed_rank", "Mann_Whitney_U_Test", "Friedman_Chi_Square", "Kruskal_Wallis"
            ]
        else:  # Categorical
            st.write("### Recommended Tests for Categorical Data")
            methods = [
                "McNemar_test", "Chi_squared_test", "Fisher_exact_test", "Cochran_Q_test", "Marginal_Homogeneity_test"
            ]

        selected_method = st.selectbox(f"Choose a Hypothesis test to perform for {column}", methods, key=column)

        if st.button(f"Run Test for {column}"):
            try:
                result_message = ""
                data_for_test = all_groups[column]

                if selected_method in ["t_test_independent", "Mann_Whitney_U_Test", "Wilcoxon_signed_rank", "dependent_ttest"]:
                    if len(data_for_test) < 2:
                        st.error("At least two groups are required for this test.")
                        continue
                    group1, group2 = data_for_test[:len(data_for_test)//2], data_for_test[len(data_for_test)//2:]

                    if selected_method == "t_test_independent":
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        result_message = f"T-Test Independent: Test Statistic = {t_stat:.4f}, P-value = {p_value:.4f}"
                    elif selected_method == "Mann_Whitney_U_Test":
                        u_stat, p_value = stats.mannwhitneyu(group1, group2)
                        result_message = f"Mann-Whitney U Test: U Statistic = {u_stat:.4f}, P-value = {p_value:.4f}"
                    elif selected_method == "Wilcoxon_signed_rank":
                        w_stat, p_value = stats.wilcoxon(group1, group2)
                        result_message = f"Wilcoxon Signed-Rank Test: W Statistic = {w_stat:.4f}, P-value = {p_value:.4f}"
                    elif selected_method == "dependent_ttest":
                        t_stat, p_value = stats.ttest_rel(group1, group2)
                        result_message = f"Dependent T-Test: Test Statistic = {t_stat:.4f}, P-value = {p_value:.4f}"

                elif selected_method in ["Chi_squared_test", "Fisher_exact_test", "McNemar_test"]:
                    contingency_table = pd.crosstab(data_for_test, data_for_test)
                    if selected_method == "Chi_squared_test":
                        chi2, p_value, _, _ = chi2_contingency(contingency_table)
                        result_message = f"Chi-Squared Test: Chi2 Statistic = {chi2:.4f}, P-value = {p_value:.4f}"
                    elif selected_method == "Fisher_exact_test":
                        odds_ratio, p_value = fisher_exact(contingency_table)
                        result_message = f"Fisher Exact Test: Odds Ratio = {odds_ratio:.4f}, P-value = {p_value:.4f}"
                    elif selected_method == "McNemar_test":
                        result = mcnemar(contingency_table, exact=False, correction=True)
                        result_message = f"McNemar Test: Statistic = {result.statistic:.4f}, P-value = {result.pvalue:.4f}"

                st.info(result_message)
                if "p_value" in locals() and p_value < 0.05:
                    st.success("Result: Reject null hypothesis")
                else:
                    st.warning("Result: Fail to reject null hypothesis")

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.write("Thank you for using the Hypothesis Testing Application!")
