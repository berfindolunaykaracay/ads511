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

    all_groups = []
    column_types = {}

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
                column_types[selected_test_col] = "Numerical"
                all_groups.append(cleaned_data.tolist())

            elif pd.api.types.is_categorical_dtype(cleaned_data):
                st.write(f"**Column '{selected_test_col}' is categorical.**")
                column_types[selected_test_col] = "Categorical"
                all_groups.append(cleaned_data.tolist())

            else:
                st.write(f"**Column '{selected_test_col}' contains unsupported data for hypothesis testing.")

    if not all_groups:
        st.warning("No valid data provided.")
        st.stop()

    # Step 3: Otomatik Veri Türü Seçimi
    if len(set(column_types.values())) > 1:
        st.warning("Selected columns have mixed data types. Please adjust your selections.")
        st.stop()

    data_type = list(column_types.values())[0]
    st.info(f"Detected data type: {data_type}")

    # Step 4: Assumption Checks (Numerical Data)
    if data_type == "Numerical":
        st.header("Step 4: Assumption Check")
        st.write("Performing Normality and Variance Homogeneity Tests")

        results = []
        for i, group in enumerate(all_groups, start=1):
            try:
                test_stat_normality, p_value_normality = stats.shapiro(np.array(group))
                is_normal = p_value_normality >= 0.05
                results.append((f"Group {i}", "Normality", p_value_normality, "Pass" if is_normal else "Fail"))
            except ValueError as e:
                st.error(f"Error with Group {i}: {e}")

        if len(all_groups) > 1:
            try:
                test_stat_var, p_value_var = stats.levene(*[np.array(g) for g in all_groups])
                is_homogeneous = p_value_var >= 0.05
                results.append(("All Groups", "Variance Homogeneity", p_value_var, "Pass" if is_homogeneous else "Fail"))
            except Exception as e:
                st.error(f"Error in Variance Homogeneity Test: {e}")

        results_df = pd.DataFrame(results, columns=["Group", "Test", "P-value", "Result"])
        st.table(results_df)

        if all(res[3] == "Pass" for res in results):
            st.info("Your data is parametric data")
            parametric = True
        else:
            st.info("Your data is non-parametric data")
            parametric = False

    # Step 5: Hypothesis Testing
    st.header("Step 5: Select and Perform a Hypothesis Test")

    if data_type == "Numerical":
        if parametric:
            methods = [
                "t_test_independent", "dependent_ttest", "repeated_measure_anova", "oneway_anova"
            ]
        else:
            methods = [
                "Wilcoxon_signed_rank", "Mann_Whitney_U_Test", "Friedman_Chi_Square", "Kruskal_Wallis"
            ]
    else:
        methods = [
            "McNemar_test", "Chi_squared_test", "Fisher_exact_test",
            "Cochran_Q_test", "Marginal_Homogeneity_test"
        ]

    selected_method = st.selectbox("Choose a Hypothesis test to perform", methods)

    if st.button("Run Test"):
        result_message = ""
        try:
            if selected_method in ["t_test_independent", "Mann_Whitney_U_Test", "Wilcoxon_signed_rank", "dependent_ttest"]:
                if len(all_groups) < 2:
                    st.error("At least two groups are required for this test.")
                else:
                    group1, group2 = all_groups[:2]
                    if selected_method == "t_test_independent":
                        ttest, p_value = stats.ttest_ind(group1, group2)
                        result_message = f"T-test Independent: Test Statistic = {ttest:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
                    elif selected_method == "Mann_Whitney_U_Test":
                        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
                        result_message = f"Mann-Whitney U Test: Test Statistic = {u_stat:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
                    elif selected_method == "Wilcoxon_signed_rank":
                        w_stat, p_value = stats.wilcoxon(group1, group2)
                        result_message = f"Wilcoxon Signed-Rank Test: Test Statistic = {w_stat:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
                    elif selected_method == "dependent_ttest":
                        t_stat, p_value = stats.ttest_rel(group1, group2)
                        result_message = f"Dependent T-test: Test Statistic = {t_stat:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")

            elif selected_method in ["oneway_anova", "Kruskal_Wallis", "Friedman_Chi_Square"]:
                if len(all_groups) < 3:
                    st.error("At least three groups are required for this test.")
                else:
                    group1, group2, group3 = all_groups[:3]
                    if selected_method == "oneway_anova":
                        f_stat, p_value = stats.f_oneway(group1, group2, group3)
                        result_message = f"One-Way ANOVA: F Statistic = {f_stat:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
                    elif selected_method == "Kruskal_Wallis":
                        h_stat, p_value = stats.kruskal(group1, group2, group3)
                        result_message = f"Kruskal-Wallis Test: H Statistic = {h_stat:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
                    elif selected_method == "Friedman_Chi_Square":
                        chi_stat, p_value = stats.friedmanchisquare(group1, group2, group3)
                        result_message = f"Friedman Chi-Square Test: Chi-Square = {chi_stat:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")

            elif selected_method in ["McNemar_test", "Chi_squared_test", "Fisher_exact_test", "Cochran_Q_test", "Marginal_Homogeneity_test"]:
                if len(all_groups) < 2:
                    st.error("Categorical tests require at least two groups.")
                else:
                    group1, group2 = all_groups[:2]
                    if selected_method == "McNemar_test":
                        table = pd.crosstab(group1, group2)
                        result, p_value = mcnemar(table, exact=False, correction=True)
                        result_message = f"McNemar Test: Statistic = {result.statistic:.4f}, P-value = {result.pvalue:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
                    elif selected_method == "Chi_squared_test":
                        table = pd.crosstab(group1, group2)
                        chi2, p_value, _, _ = chi2_contingency(table)
                        result_message = f"Chi-Squared Test: Chi2 Statistic = {chi2:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
                    elif selected_method == "Fisher_exact_test":
                        table = pd.crosstab(group1, group2)
                        odds_ratio, p_value = fisher_exact(table)
                        result_message = f"Fisher Exact Test: Odds Ratio = {odds_ratio:.4f}, P-value = {p_value:.4f}"
                        st.info(result_message)
                        if p_value < 0.05:
                            st.success("Result: Reject null hypothesis")
                        else:
                            st.warning("Result: Fail to reject null hypothesis")
        except Exception as e:
            st.warning(f"You may have chosen the wrong hypothesis test. Please check Hypothesis Testing Map.")
            st.error(f"Error: {e}")

st.write("Thank you for using the Hypothesis Testing Application!")
