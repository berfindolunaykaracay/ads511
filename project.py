import streamlit as st
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, f_oneway, mannwhitneyu, kruskal, wilcoxon, chi2_contingency

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

    if selected_columns:
        for selected_test_col in selected_columns:
            st.markdown(f"<h3 style='text-align: center;'>Selected Column: {selected_test_col}</h3>", unsafe_allow_html=True)

            cleaned_data = data[selected_test_col].dropna()
            if cleaned_data.empty:
                st.error(f"Column '{selected_test_col}' has no valid data for hypothesis testing.")
                continue

            # Veri Türü Tespiti
            if pd.api.types.is_numeric_dtype(cleaned_data):
                st.write(f"**Column '{selected_test_col}' is numeric.**")

                # Grup Sayısı ve Bağımlılık Durumu Seçimi
                group_count = st.radio(f"Number of Groups for {selected_test_col}", options=["2 Groups", ">2 Groups"], key=f"group_{selected_test_col}")
                dependency = st.radio(f"Dependency for {selected_test_col}", options=["Dependent", "Independent"], key=f"dependency_{selected_test_col}")

                # Test Önerileri
                st.markdown("### Suggested Tests")
                if group_count == "2 Groups":
                    if dependency == "Dependent":
                        test_options = ["Paired T-Test", "Wilcoxon Signed-Rank Test"]
                    else:
                        test_options = ["Student T-Test", "Mann-Whitney U Test"]
                elif group_count == ">2 Groups":
                    if dependency == "Dependent":
                        test_options = ["Repeated Measures ANOVA", "Friedman Test"]
                    else:
                        test_options = ["One-Way ANOVA", "Kruskal-Wallis Test"]

            elif pd.api.types.is_categorical_dtype(cleaned_data):
                st.write(f"**Column '{selected_test_col}' is categorical.**")

                # Grup Sayısı ve Bağımlılık Durumu Seçimi
                group_count = st.radio(f"Number of Groups for {selected_test_col}", options=["2 Groups", ">2 Groups"], key=f"group_{selected_test_col}_cat")
                dependency = st.radio(f"Dependency for {selected_test_col}", options=["Dependent", "Independent"], key=f"dependency_{selected_test_col}_cat")

                # Test Önerileri
                st.markdown("### Suggested Tests")
                if group_count == "2 Groups":
                    if dependency == "Dependent":
                        test_options = ["McNemar Test", "Fisher Exact Test"]
                    else:
                        test_options = ["Chi-Square Test"]
                elif group_count == ">2 Groups":
                    if dependency == "Dependent":
                        test_options = ["Marginal Homogeneity Test"]
                    else:
                        test_options = ["Chi-Square Test"]

            elif pd.api.types.is_object_dtype(cleaned_data):
                st.write(f"**Column '{selected_test_col}' contains text data, which is not suitable for hypothesis testing.**")
                test_options = []

            else:
                st.write(f"**Column '{selected_test_col}' is not supported for hypothesis testing.**")
                test_options = []

            # Test Seçimi
            selected_test = st.radio(f"Select the test to perform for {selected_test_col}", test_options, key=f"test_{selected_test_col}")

            # Step 3: Testin Gerçekleştirilmesi
            if st.button(f"Run Test for {selected_test_col}"):
                try:
                    if selected_test == "Paired T-Test":
                        group1 = cleaned_data[:-1]
                        group2 = cleaned_data[1:]
                        t_stat, p_val = ttest_rel(group1, group2)
                        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                    elif selected_test == "Student T-Test":
                        group_column = st.selectbox("Select column to define groups", data.columns)
                        unique_values = data[group_column].dropna().unique()
                        if len(unique_values) < 2:
                            st.error("Not enough groups for Student T-Test.")
                            continue
                        group1 = cleaned_data[data[group_column] == unique_values[0]]
                        group2 = cleaned_data[data[group_column] == unique_values[1]]
                        t_stat, p_val = ttest_ind(group1, group2)
                        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                    elif selected_test == "Mann-Whitney U Test":
                        group1 = cleaned_data[:len(cleaned_data)//2]
                        group2 = cleaned_data[len(cleaned_data)//2:]
                        u_stat, p_val = mannwhitneyu(group1, group2)
                        st.write(f"U-Statistic: {u_stat}, P-Value: {p_val}")
                    elif selected_test == "Chi-Square Test":
                        comparison_column = st.selectbox("Select second column for Chi-Square test", data.columns)
                        contingency_table = pd.crosstab(cleaned_data, data[comparison_column])
                        chi2, p_val, _, _ = chi2_contingency(contingency_table)
                        st.write(f"Chi2 Statistic: {chi2}, P-Value: {p_val}")
                except Exception as e:
                    st.error(f"An error occurred while running the test: {e}")
else:
    st.info("Please upload a dataset to proceed.")
