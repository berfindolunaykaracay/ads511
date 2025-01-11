import streamlit as st
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, f_oneway, wilcoxon, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency, fisher_exact

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Application</h1>",
            unsafe_allow_html=True)

# Step 1: Veri Yükleme
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 1: Upload Dataset</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center;'>Dataset Preview</h2>", unsafe_allow_html=True)
    st.dataframe(data.head())

    # Step 2: Sütun Seçimi
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Column</h2>", unsafe_allow_html=True)
    selected_columns = st.multiselect("Select columns for hypothesis testing", data.columns.tolist())

    if selected_columns:
        selected_test_col = st.selectbox("Select a column for hypothesis testing", selected_columns)

        if selected_test_col:
            # Step 3: Test Seçimi
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 3: Select Groups and Dependency</h2>",
                        unsafe_allow_html=True)

            # Veri türüne göre yönlendirme
            if pd.api.types.is_numeric_dtype(data[selected_test_col]):
                st.write(f"**Column '{selected_test_col}' is numeric.**")
                group_count = st.radio("Number of Groups", options=["2 Groups", ">2 Groups"])
                dependency = st.radio("Dependency", options=["Dependent", "Independent"])

                # Sayısal Veri - 2 Grup
                if group_count == "2 Groups":
                    if dependency == "Dependent":
                        test_options = ["Paired T-Test", "Wilcoxon Signed-Rank Test"]
                    else:
                        test_options = ["Student T-Test", "Mann-Whitney U Test"]

                # Sayısal Veri - >2 Grup
                elif group_count == ">2 Groups":
                    if dependency == "Dependent":
                        test_options = ["Repeated Measures ANOVA", "Friedman Test"]
                    else:
                        test_options = ["One-Way ANOVA", "Kruskal-Wallis Test"]

            elif pd.api.types.is_categorical_dtype(data[selected_test_col]) or data[selected_test_col].dtype == object:
                st.write(f"**Column '{selected_test_col}' is categorical.**")
                unique_values = data[selected_test_col].nunique()
                if unique_values == 2:
                    group_count = "2 Categories"
                else:
                    group_count = ">2 Categories"

                if group_count == "2 Categories":
                    dependency = st.radio("Dependency", options=["Dependent", "Independent"])
                    if dependency == "Dependent":
                        test_options = ["McNemar Test", "Fisher Exact Test"]
                    else:
                        test_options = ["Chi-Square Test"]
                elif group_count == ">2 Categories":
                    dependency = st.radio("Dependency", options=["Dependent", "Independent"])
                    if dependency == "Dependent":
                        test_options = ["Marginal Homogeneity Test"]
                    else:
                        test_options = ["Chi-Square Test"]

            else:
                st.error("Unsupported column type. Please select another column.")
                test_options = []

            # Test Seçimi
            if test_options:
                selected_test = st.radio("Select the test to perform", test_options)

                # Step 4: Testin Gerçekleştirilmesi
                st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 4: Run Hypothesis Test</h2>",
                            unsafe_allow_html=True)
                if st.button("Run Test"):
                    try:
                        if selected_test == "Paired T-Test":
                            col_data = data[selected_test_col].dropna()
                            t_stat, p_val = ttest_rel(col_data[:-1], col_data[1:])
                            st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                        elif selected_test == "Student T-Test":
                            col_data = data[selected_test_col].dropna()
                            group1 = col_data[:len(col_data) // 2]
                            group2 = col_data[len(col_data) // 2:]
                            t_stat, p_val = ttest_ind(group1, group2)
                            st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                        elif selected_test == "One-Way ANOVA":
                            col_data = data[selected_test_col].dropna()
                            groups = [col_data[:len(col_data) // 3],
                                      col_data[len(col_data) // 3:2 * len(col_data) // 3],
                                      col_data[2 * len(col_data) // 3:]]
                            f_stat, p_val = f_oneway(*groups)
                            st.write(f"F-Statistic: {f_stat}, P-Value: {p_val}")
                        elif selected_test == "Chi-Square Test":
                            contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                            chi2, p_val, _, _ = chi2_contingency(contingency_table)
                            st.write(f"Chi2 Statistic: {chi2}, P-Value: {p_val}")
                        elif selected_test == "McNemar Test":
                            contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                            _, p_val = fisher_exact(contingency_table.values)
                            st.write(f"P-Value: {p_val}")
                        elif selected_test == "Wilcoxon Signed-Rank Test":
                            col_data = data[selected_test_col].dropna()
                            stat, p_val = wilcoxon(col_data[:-1], col_data[1:])
                            st.write(f"Wilcoxon Statistic: {stat}, P-Value: {p_val}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to proceed.")
