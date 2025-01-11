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
                result_message = ""
                try:
                    if selected_test in ["Paired T-Test", "Wilcoxon Signed-Rank Test", "Student T-Test", "Mann-Whitney U Test"]:
                        if len(cleaned_data) < 2:
                            st.error("At least two groups are required for this test.")
                        else:
                            group1, group2 = cleaned_data[:len(cleaned_data)//2], cleaned_data[len(cleaned_data)//2:]
                            if selected_test == "Paired T-Test":
                                t_stat, p_value = ttest_rel(group1, group2)
                                result_message = f"Paired T-Test: T-Statistic = {t_stat:.4f}, P-value = {p_value:.4f}"
                            elif selected_test == "Student T-Test":
                                t_stat, p_value = ttest_ind(group1, group2)
                                result_message = f"Student T-Test: T-Statistic = {t_stat:.4f}, P-value = {p_value:.4f}"
                            elif selected_test == "Mann-Whitney U Test":
                                u_stat, p_value = mannwhitneyu(group1, group2)
                                result_message = f"Mann-Whitney U Test: U-Statistic = {u_stat:.4f}, P-value = {p_value:.4f}"
                            elif selected_test == "Wilcoxon Signed-Rank Test":
                                w_stat, p_value = wilcoxon(group1, group2)
                                result_message = f"Wilcoxon Signed-Rank Test: W-Statistic = {w_stat:.4f}, P-value = {p_value:.4f}"
                    elif selected_test in ["One-Way ANOVA", "Kruskal-Wallis Test", "Friedman Test"]:
                        if len(cleaned_data) < 3:
                            st.error("At least three groups are required for this test.")
                        else:
                            group1, group2, group3 = cleaned_data[:len(cleaned_data)//3], cleaned_data[len(cleaned_data)//3:2*len(cleaned_data)//3], cleaned_data[2*len(cleaned_data)//3:]
                            if selected_test == "One-Way ANOVA":
                                f_stat, p_value = f_oneway(group1, group2, group3)
                                result_message = f"One-Way ANOVA: F-Statistic = {f_stat:.4f}, P-value = {p_value:.4f}"
                            elif selected_test == "Kruskal-Wallis Test":
                                h_stat, p_value = kruskal(group1, group2, group3)
                                result_message = f"Kruskal-Wallis Test: H-Statistic = {h_stat:.4f}, P-value = {p_value:.4f}"
                            elif selected_test == "Friedman Test":
                                chi_stat, p_value = kruskal(group1, group2, group3)  # Replace with correct Friedman test if available
                                result_message = f"Friedman Test: Chi-Statistic = {chi_stat:.4f}, P-value = {p_value:.4f}"
                    elif selected_test in ["Chi-Square Test", "Fisher Exact Test"]:
                        contingency_table = pd.crosstab(group1, group2)
                        if selected_test == "Chi-Square Test":
                            chi2, p_value, _, _ = chi2_contingency(contingency_table)
                            result_message = f"Chi-Square Test: Chi2 Statistic = {chi2:.4f}, P-value = {p_value:.4f}"
                        elif selected_test == "Fisher Exact Test":
                            odds_ratio, p_value = fisher_exact(contingency_table)
                            result_message = f"Fisher Exact Test: Odds Ratio = {odds_ratio:.4f}, P-value = {p_value:.4f}"
                    if result_message:
                        st.info(result_message)
                except Exception as e:
                    st.error(f"An error occurred while running the test: {e}")
else:
    st.info("Please upload a dataset to proceed.")
