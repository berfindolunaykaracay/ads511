import streamlit as st
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, f_oneway, wilcoxon, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Application</h1>", unsafe_allow_html=True)

# Step 1: Veri Yükleme
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 1: Upload Your Dataset</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center;'>Dataset Preview</h2>", unsafe_allow_html=True)
    st.dataframe(data.head())

    # Step 2: Sütun Seçimi
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Columns</h2>", unsafe_allow_html=True)
    selected_columns = st.multiselect("Select columns for hypothesis testing", data.columns.tolist())

    if selected_columns:
        st.markdown("<h3 style='text-align: center;'>Selected Columns</h3>", unsafe_allow_html=True)
        st.write(data[selected_columns].head())

        # Step 3: Test Türü Belirleme
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 3: Select Hypothesis Test</h2>", unsafe_allow_html=True)

        # Veri Türüne Göre Test Seçeneklerini Hazırlama
        test_options = {}
        for col in selected_columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                test_options[col] = ["Paired T-Test", "Independent T-Test", "One-Way ANOVA"]
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == object:
                unique_values = data[col].nunique()
                if unique_values == 2:
                    test_options[col] = ["Chi-Square Test", "McNemar Test", "Fisher Exact Test"]
                elif unique_values > 2:
                    test_options[col] = ["Chi-Square Test"]

        # Sütun ve Test Seçimi
        selected_test_col = st.selectbox("Select a column for testing", list(test_options.keys()))
        if selected_test_col:
            available_tests = test_options[selected_test_col]
            selected_test = st.selectbox("Choose a hypothesis test to perform", available_tests)

            # Step 4: Testi Gerçekleştirme
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 4: Run Hypothesis Test</h2>", unsafe_allow_html=True)
            if st.button("Run Test"):
                try:
                    if selected_test == "Paired T-Test":
                        col_data = data[selected_test_col].dropna()
                        t_stat, p_val = ttest_rel(col_data[:-1], col_data[1:])
                        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                    elif selected_test == "Independent T-Test":
                        col_data = data[selected_test_col].dropna()
                        group1 = col_data[:len(col_data)//2]
                        group2 = col_data[len(col_data)//2:]
                        t_stat, p_val = ttest_ind(group1, group2)
                        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                    elif selected_test == "One-Way ANOVA":
                        col_data = data[selected_test_col].dropna()
                        groups = [col_data[:len(col_data)//3], col_data[len(col_data)//3:2*len(col_data)//3], col_data[2*len(col_data)//3:]]
                        f_stat, p_val = f_oneway(*groups)
                        st.write(f"F-Statistic: {f_stat}, P-Value: {p_val}")
                    elif selected_test == "Chi-Square Test":
                        contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                        chi2, p_val, _, _ = chi2_contingency(contingency_table)
                        st.write(f"Chi2 Statistic: {chi2}, P-Value: {p_val}")
                    elif selected_test == "McNemar Test":
                        contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                        _, p_val = proportions_ztest(contingency_table.iloc[0], contingency_table.iloc[1])
                        st.write(f"P-Value: {p_val}")
                    elif selected_test == "Fisher Exact Test":
                        contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                        _, p_val = fisher_exact(contingency_table.values)
                        st.write(f"P-Value: {p_val}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to proceed.")
