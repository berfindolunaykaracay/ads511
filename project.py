import streamlit as st
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, f_oneway, wilcoxon, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Guide</h1>", unsafe_allow_html=True)

# Step 1: CSV Dosyası Yükleme
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 1: Upload a CSV file</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Dataset Preview</h2>", unsafe_allow_html=True)

    # Dataset bilgisi
    st.write(f"**Number of rows:** {data.shape[0]}")
    st.write(f"**Number of columns:** {data.shape[1]}")
    st.write("### Full Dataset")
    st.dataframe(data)

    # Step 2: Sütun Seçimi
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Columns</h2>", unsafe_allow_html=True)
    selected_columns = st.multiselect("Select columns", data.columns.tolist())

    if selected_columns:
        st.write("### Selected Columns Data")
        st.dataframe(data[selected_columns].reset_index(drop=True))

        # Step 3: Hipotez testi için sütun seçimi
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 3: Select Columns for Hypothesis Testing</h2>", unsafe_allow_html=True)
        testing_columns = st.multiselect("Select columns for testing", selected_columns)

        if testing_columns:
            st.write(f"### Selected Columns for Testing: {', '.join(testing_columns)}")

            # Step 4: Veri türü tespiti ve test önerileri
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 4: Recommended Tests Based on Data Type</h2>", unsafe_allow_html=True)

            test_options = {}
            for col in testing_columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    unique_values = data[col].nunique()
                    if unique_values == 2:
                        dependency = "Dependent" if data[col].duplicated().any() else "Independent"
                        if dependency == "Dependent":
                            test_options[col] = ["Paired T-Test", "Wilcoxon Signed-Rank Test"]
                        else:
                            test_options[col] = ["Independent T-Test", "Mann-Whitney U Test"]
                    else:
                        if len(data[col].dropna().unique()) > 2:
                            test_options[col] = ["One-Way ANOVA", "Kruskal-Wallis Test"]
                elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == object:
                    unique_values = data[col].nunique()
                    if unique_values == 2:
                        dependency = "Dependent" if data[col].duplicated().any() else "Independent"
                        if dependency == "Dependent":
                            test_options[col] = ["McNemar Test", "Fisher Exact Test"]
                        else:
                            test_options[col] = ["Chi-Square Test"]
                    elif unique_values > 2:
                        dependency = "Independent"
                        test_options[col] = ["Chi-Square Test", "Cochran's Q Test"]

            # Kullanıcıya sütun ve test seçme seçenekleri
            selected_test_col = st.selectbox("Choose a column for testing", list(test_options.keys()))
            if selected_test_col:
                available_tests = test_options[selected_test_col]
                selected_test = st.selectbox("Choose a Hypothesis Test to Perform", available_tests)

                # Test gerçekleştirme
                if st.button("Run Test"):
                    st.markdown(f"<h3 style='text-align: center;'>Performing: {selected_test} on {selected_test_col}</h3>", unsafe_allow_html=True)
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
                        elif selected_test == "Wilcoxon Signed-Rank Test":
                            col_data = data[selected_test_col].dropna()
                            stat, p_val = wilcoxon(col_data[:-1], col_data[1:])
                            st.write(f"Wilcoxon Statistic: {stat}, P-Value: {p_val}")
                        elif selected_test == "Fisher Exact Test":
                            contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                            _, p_val = fisher_exact(contingency_table.values)
                            st.write(f"P-Value: {p_val}")
                    except Exception as e:
                        st.write(f"Error while performing the test: {e}")
else:
    st.write("No columns selected for testing.")
