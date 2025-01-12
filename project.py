import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency, fisher_exact
pd.options.display.float_format = '{:,.4f}'.format

def check_normality(data):
    if not isinstance(data, (list, np.ndarray, pd.Series)):
        raise ValueError("Data must be a list, NumPy array, or pandas Series.")
    data = np.array(data)

    test_stat_normality, p_value_normality = stats.shapiro(data)
    return p_value_normality, p_value_normality >= 0.05

def check_variance_homogeneity(data):
    if len(data) < 2:
        return None, False
    test_stat_var, p_value_var = stats.levene(*data)
    return p_value_var, p_value_var >= 0.05

# Test descriptions
numerical_tests_parametric = {
    "Independent T-Test": "Examines if two independent groups have significantly different means, assuming data normality and equal variances.",
    "Dependent (Paired) T-Test": "Analyzes mean differences between two related groups, assuming normal distribution of paired differences.",
    "Repeated Measures ANOVA": "Identifies significant differences across multiple measurements for the same group under different conditions.",
    "One-Way ANOVA": "Determines whether there are significant differences in means among three or more independent groups."
}

numerical_tests_nonparametric = {
    "Wilcoxon Signed-Rank Test": "Tests for differences in medians of paired samples, suitable for non-normal data.",
    "Mann-Whitney U Test": "Compares distributions of two independent groups without assuming normality.",
    "Friedman Test (Chi-Square)": "Detects differences across repeated measures in non-parametric settings.",
    "Kruskal-Wallis Test": "Assesses differences in medians among three or more independent groups, ideal for ordinal data or non-normal distributions."
}

categorical_tests = {
    "McNemar Test": "Evaluates changes in paired categorical data, commonly used for pre- and post-intervention comparisons.",
    "Chi-Squared Test": "Tests for associations between two categorical variables using contingency tables.",
    "Fisher's Exact Test": "Analyzes relationships in small sample categorical data, providing exact p-values.",
    "Cochran Q Test": "Assesses differences in proportions across multiple related groups in binary data.",
    "Marginal Homogeneity Test": "Examines shifts in paired categorical data to detect significant changes."
}

# Streamlit app starts here
st.set_page_config(page_title="Hypothesis Testing Application", page_icon="⚛", layout="wide")

# Sidebar header and logo
try:
    st.sidebar.image("TEDU_LOGO.png", use_container_width=True)
except Exception:
    st.sidebar.warning("Logo file not found. Please check the file path.")

st.sidebar.title("ADS 511: Statistical Inference Methods")
st.sidebar.write("Developed by: Serdar Hosver")

st.sidebar.title("Hypothesis Testing Map")
try:
    st.sidebar.image("Hypothesis_Test_Map.png", use_container_width=True)
except Exception:
    st.sidebar.warning("Hypothesis Testing Map image not found. Please check the file path.")

# Sidebar to display all available tests
st.sidebar.header("List of Available Hypothesis Tests")
for category, tests in [("Parametric Tests", numerical_tests_parametric),
                        ("Non-Parametric Tests", numerical_tests_nonparametric),
                        ("Categorical Tests", categorical_tests)]:
    st.sidebar.subheader(category)
    for test, description in tests.items():
        with st.sidebar.expander(f"ℹ️ {test}"):
            st.write(description)

# Title and Introduction
st.markdown("<h1 style='text-align: center;'>Hypothesis Testing Application</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This application helps you conduct various <b>hypothesis tests</b>. Simply upload your data and proceed with guided testing steps.</p>", unsafe_allow_html=True)

# Data Input
st.markdown("<h2 style='text-align: center;'>Data Input</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

all_groups = []
column_names = []
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    num_rows, num_cols = data.shape
    st.write(f"### Dataset Information:")
    st.write(f"The dataset contains **{num_rows} rows** and **{num_cols} columns**.")

    st.write("### Full Dataset Preview:")
    st.write(data)

    columns = st.radio("Choose column selection method:", ("Select All Columns", "Manually Select Columns"))

    if columns == "Select All Columns":
        column_names = data.select_dtypes(include=[np.number]).columns.tolist()
        all_groups = [data[col].dropna().tolist() for col in column_names]
        st.write("### Selected Columns Preview:")
        st.write(data[column_names])

    elif columns == "Manually Select Columns":
        selected_columns = st.multiselect("Pick columns to include", options=data.columns)
        if selected_columns:
            column_names = [col for col in selected_columns if pd.api.types.is_numeric_dtype(data[col])]
            all_groups = [data[col].dropna().tolist() for col in column_names]
            st.write("### Selected Columns Preview:")
            st.write(data[column_names])

if not all_groups:
    st.warning("No valid data provided.")
    st.stop()

# Data Type Selection
data_type = st.selectbox(
    "Choose your data type:",
    options=["Numerical Data", "Categorical Data"]
)

# Assumption Checks
if data_type == "Numerical Data":
    st.markdown("<h2 style='text-align: center;'>Assumption Checks</h2>", unsafe_allow_html=True)
    st.write("Verifying Normality and Variance Homogeneity Assumptions")

    for col_name, group in zip(column_names, all_groups):
        try:
            p_normality, is_normal = check_normality(group)
            if is_normal:
                st.success(f"Normality Check for {col_name}: P-value = {p_normality:.4f} (Pass)")
            else:
                st.warning(f"Normality Check for {col_name}: P-value = {p_normality:.4f} (Fail)")
        except ValueError as e:
            st.error(f"Error with column {col_name}: {e}")

    if len(all_groups) > 1:
        try:
            p_variance, is_homogeneous = check_variance_homogeneity(all_groups)
            if is_homogeneous:
                st.success(f"Variance Homogeneity Check: P-value = {p_variance:.4f} (Pass)")
            else:
                st.warning(f"Variance Homogeneity Check: P-value = {p_variance:.4f} (Fail)")
        except Exception as e:
            st.error(f"Error in Variance Homogeneity Check: {e}")

    parametric = all(check_normality(group)[1] for group in all_groups)
    if parametric:
        st.info("The data meets parametric assumptions.")
    else:
        st.info("The data does not meet parametric assumptions.")

    # Recommended Test
    if parametric and len(all_groups) == 2:
        st.success("Recommended Test: Independent T-Test or Dependent T-Test (for paired groups).")
    elif parametric and len(all_groups) > 2:
        st.success("Recommended Test: One-Way ANOVA.")
    elif not parametric and len(all_groups) == 2:
        st.success("Recommended Test: Mann-Whitney U Test or Wilcoxon Signed-Rank Test (for paired groups).")
    elif not parametric and len(all_groups) > 2:
        st.success("Recommended Test: Kruskal-Wallis Test or Friedman Test (for repeated measures).")
    else:
        st.info("Ensure data compatibility with selected test.")

# Hypothesis Testing
st.markdown("<h2 style='text-align: center;'>Hypothesis Testing</h2>", unsafe_allow_html=True)

if data_type == "Numerical Data":
    test_list = numerical_tests_parametric if parametric else numerical_tests_nonparametric
else:
    test_list = categorical_tests

selected_test = st.selectbox("Choose the test you want to perform:", list(test_list.keys()))

# Display test description
st.info(f"**Test Overview:** {test_list[selected_test]}")

if st.button("Run Test"):
    try:
        if selected_test == "Independent T-Test" and len(all_groups) >= 2:
            t_stat, p_value = stats.ttest_ind(all_groups[0], all_groups[1])
            st.success(f"Independent T-Test Results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        elif selected_test == "Dependent (Paired) T-Test" and len(all_groups) >= 2:
            t_stat, p_value = stats.ttest_rel(all_groups[0], all_groups[1])
            st.success(f"Dependent T-Test Results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        elif selected_test == "One-Way ANOVA" and len(all_groups) > 2:
            f_stat, p_value = stats.f_oneway(*all_groups)
            st.success(f"One-Way ANOVA Results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
        elif selected_test == "Mann-Whitney U Test" and len(all_groups) >= 2:
            u_stat, p_value = stats.mannwhitneyu(all_groups[0], all_groups[1])
            st.success(f"Mann-Whitney U Test Results: U-statistic = {u_stat:.4f}, p-value = {p_value:.4f}")
        else:
            st.error("The selected test is not implemented or requires more groups.")
    except Exception as e:
        st.error(f"An error occurred during the test: {e}")

st.write("Thank you for using the Hypothesis Testing Application!")
