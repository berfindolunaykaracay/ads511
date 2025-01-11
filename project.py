import streamlit as st
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, chi2_contingency, f_oneway
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
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 4: Recommended Tests</h2>", unsafe_allow_html=True)
            recommendations = []

            for col in testing_columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    unique_values = data[col].nunique()
                    if unique_values <= 10:  # Küçük grup sayısı için kontrol
                        dependency = "Dependent" if data[col].duplicated().any() else "Independent"
                        if dependency == "Dependent":
                            recommendations.append((col, "Paired T-Test"))
                        else:
                            recommendations.append((col, "Independent T-Test"))
                    else:
                        dependency = "Independent"  # Çok fazla grup olduğunda bağımsız kabul edilir
                        recommendations.append((col, "One-Way ANOVA"))
                else:
                    unique_values = data[col].nunique()
                    if unique_values == 2:
                        dependency = "Dependent" if data[col].duplicated().any() else "Independent"
                        if dependency == "Dependent":
                            recommendations.append((col, "McNemar Test"))
                        else:
                            recommendations.append((col, "Chi-Square Test"))
                    elif unique_values > 2:
                        dependency = "Independent"  # Çok kategorili bağımsız kabul edilir
                        recommendations.append((col, "Chi-Square Test"))

            if recommendations:
                for col, test in recommendations:
                    st.write(f"- **{col}:** {test}")
            else:
                st.write("No suitable tests found for the selected configuration.")

            # Step 5: Test Gerçekleştirme
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 5: Perform a Hypothesis Test</h2>", unsafe_allow_html=True)
            selected_test_col = st.selectbox("Choose a column for testing", [col for col, _ in recommendations])
            selected_test = st.selectbox("Choose a Hypothesis Test to Perform", [test for _, test in recommendations])

            if st.button("Run Test"):
                st.markdown(f"<h3 style='text-align: center;'>Performing: {selected_test} on {selected_test_col}</h3>", unsafe_allow_html=True)
                try:
                    if selected_test == "Paired T-Test":
                        col_data = data[selected_test_col].dropna()
                        if len(col_data) < 2:
                            st.error("Not enough data for Paired T-Test. At least 2 observations are required.")
                        else:
                            t_stat, p_val = ttest_rel(col_data[:-1], col_data[1:])
                            st.success(f"T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4f}")
                    elif selected_test == "Independent T-Test":
                        col_data = data[selected_test_col].dropna()
                        if len(col_data) < 2:
                            st.error("Not enough data for Independent T-Test. At least 2 groups are required.")
                        else:
                            group1 = col_data[:len(col_data)//2]
                            group2 = col_data[len(col_data)//2:]
                            t_stat, p_val = ttest_ind(group1, group2)
                            st.success(f"T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4f}")
                    elif selected_test == "One-Way ANOVA":
                        col_data = data[selected_test_col].dropna()
                        if len(col_data) < 3:
                            st.error("Not enough data for One-Way ANOVA. At least 3 groups are required.")
                        else:
                            groups = [col_data[:len(col_data)//3], col_data[len(col_data)//3:2*len(col_data)//3], col_data[2*len(col_data)//3:]]
                            f_stat, p_val = f_oneway(*groups)
                            st.success(f"F-Statistic: {f_stat:.4f}, P-Value: {p_val:.4f}")
                    elif selected_test == "Chi-Square Test":
                        contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                            st.error("Not enough data for Chi-Square Test. Contingency table must have at least 2 rows and 2 columns.")
                        else:
                            chi2, p_val, _, _ = chi2_contingency(contingency_table)
                            st.success(f"Chi2 Statistic: {chi2:.4f}, P-Value: {p_val:.4f}")
                    elif selected_test == "McNemar Test":
                        contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                            st.error("Not enough data for McNemar Test. Contingency table must have at least 2 rows and 2 columns.")
                        else:
                            _, p_val = proportions_ztest(contingency_table.iloc[0], contingency_table.iloc[1])
                            st.success(f"P-Value: {p_val:.4f}")
                except Exception as e:
                    st.error(f"Error while performing the test: {e}")
        else:
            st.write("No columns selected for testing.")
    else:
        st.write("No columns selected.")
