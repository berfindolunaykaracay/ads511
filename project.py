import streamlit as st
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, f_oneway, mannwhitneyu, kruskal, wilcoxon, chi2_contingency, fisher_exact

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Application</h1>",
            unsafe_allow_html=True)

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
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Columns and Define Groups</h2>",
                unsafe_allow_html=True)
    selected_columns = st.multiselect("Select columns for hypothesis testing", data.columns.tolist())

    if selected_columns:
        for selected_test_col in selected_columns:
            st.markdown(f"<h3 style='text-align: center;'>Selected Column: {selected_test_col}</h3>",
                        unsafe_allow_html=True)

            # Veri Türü Tespiti
            if pd.api.types.is_numeric_dtype(data[selected_test_col]):
                st.write(f"**Column '{selected_test_col}' is numeric.**")

                # Grup Sayısı ve Bağımlılık Durumu Seçimi
                group_count = st.radio(f"Number of Groups for {selected_test_col}", options=["2 Groups", ">2 Groups"],
                                       key=f"group_{selected_test_col}")
                dependency = st.radio(f"Dependency for {selected_test_col}", options=["Dependent", "Independent"],
                                      key=f"dependency_{selected_test_col}")

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

            elif pd.api.types.is_categorical_dtype(data[selected_test_col]) or data[selected_test_col].dtype == object:
                st.write(f"**Column '{selected_test_col}' is categorical.**")

                # Grup Sayısı ve Bağımlılık Durumu Seçimi
                group_count = st.radio(f"Number of Groups for {selected_test_col}", options=["2 Groups", ">2 Groups"],
                                       key=f"group_{selected_test_col}_cat")
                dependency = st.radio(f"Dependency for {selected_test_col}", options=["Dependent", "Independent"],
                                      key=f"dependency_{selected_test_col}_cat")

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

            else:
                st.write(f"**Column '{selected_test_col}' is not supported for hypothesis testing.**")
                test_options = []

            # Test Seçimi
            selected_test = st.radio(f"Select the test to perform for {selected_test_col}", test_options,
                                     key=f"test_{selected_test_col}")

            # Step 3: Testin Gerçekleştirilmesi
            if st.button(f"Run Test for {selected_test_col}"):
                try:
                    if selected_test == "Paired T-Test":
                        t_stat, p_val = ttest_rel(data[selected_test_col].dropna()[:-1],
                                                  data[selected_test_col].dropna()[1:])
                        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                    elif selected_test == "Student T-Test":
                        group1 = data[selected_test_col].dropna()[:len(data) // 2]
                        group2 = data[selected_test_col].dropna()[len(data) // 2:]
                        t_stat, p_val = ttest_ind(group1, group2)
                        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                    elif selected_test == "Mann-Whitney U Test":
                        u_stat, p_val = mannwhitneyu(data[selected_test_col].dropna()[:len(data) // 2],
                                                     data[selected_test_col].dropna()[len(data) // 2:])
                        st.write(f"U-Statistic: {u_stat}, P-Value: {p_val}")
                    elif selected_test == "Chi-Square Test":
                        contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                        chi2, p_val, _, _ = chi2_contingency(contingency_table)
                        st.write(f"Chi2 Statistic: {chi2}, P-Value: {p_val}")
                except Exception as e:
                    st.error(f"An error occurred while running the test: {e}")
else:
    st.info("Please upload a dataset to proceed.")
