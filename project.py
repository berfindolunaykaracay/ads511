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
    "Independent T-Test": "Compares means of two independent groups. Assumes normality and equal variances.",
    "Dependent (Paired) T-Test": "Compares means of two related groups. Assumes normality of the differences.",
    "Repeated Measures ANOVA": "Tests differences across multiple measurements on the same group.",
    "One-Way ANOVA": "Tests mean differences across three or more independent groups."
}

numerical_tests_nonparametric = {
    "Wilcoxon Signed-Rank Test": "Non-parametric alternative to paired T-Test. Compares medians of two related samples.",
    "Mann-Whitney U Test": "Non-parametric alternative to independent T-Test. Compares two independent distributions.",
    "Friedman Test (Chi-Square)": "Non-parametric test for repeated measures on the same group.",
    "Kruskal-Wallis Test": "Non-parametric alternative to One-Way ANOVA. Tests differences across multiple groups."
}

categorical_tests = {
    "McNemar Test": "Tests changes in matched categorical data (e.g., before and after).",
    "Chi-Squared Test": "Tests association between two categorical variables.",
    "Fisher's Exact Test": "Exact test for association in small categorical tables.",
    "Cochran Q Test": "Tests differences in proportions across multiple matched groups.",
    "Marginal Homogeneity Test": "Tests shifts in matched categorical data."
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
st.markdown("<p style='text-align: center;'>This app allows you to perform various <b>hypothesis tests</b> with ease. Simply upload your data, and let the app guide you through hypothesis testing.</p>", unsafe_allow_html=True)

# Step 1: Data Input
st.markdown("<h2 style='text-align: center;'>Step 1: Data Input</h2>", unsafe_allow_html=True)
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

    columns = st.radio("Select one of the following options:", ("Select All Columns", "Manually Select Columns"))

    if columns == "Select All Columns":
        column_names = data.select_dtypes(include=[np.number]).columns.tolist()
        all_groups = [data[col].dropna().tolist() for col in column_names]
        st.write("### Selected Columns Preview:")
        st.write(data[column_names])

    elif columns == "Manually Select Columns":
        selected_columns = st.multiselect("Choose columns to include", options=data.columns)
        if selected_columns:
            column_names = [col for col in selected_columns if pd.api.types.is_numeric_dtype(data[col])]
            all_groups = [data[col].dropna().tolist() for col in column_names]
            st.write("### Selected Columns Preview:")
            st.write(data[column_names])

if not all_groups:
    st.warning("No valid data provided.")
    st.stop()

# Step 1.5: Data Type Selection
data_type = st.selectbox(
    "Select your data type:",
    options=["Numerical Data", "Categorical Data"]
)

# Step 2: Assumption Checks (if Numerical Data)
if data_type == "Numerical Data":
    st.markdown("<h2 style='text-align: center;'>Step 2: Assumption Check</h2>", unsafe_allow_html=True)
    st.write("Performing Normality and Variance Homogeneity Tests")

    for col_name, group in zip(column_names, all_groups):
        try:
            p_normality, is_normal = check_normality(group)
            if is_normal:
                st.success(f"Normality Test for {col_name}: P-value = {p_normality:.4f} (Pass)")
            else:
                st.warning(f"Normality Test for {col_name}: P-value = {p_normality:.4f} (Fail)")
        except ValueError as e:
            st.error(f"Error with column {col_name}: {e}")

    if len(all_groups) > 1:
        try:
            p_variance, is_homogeneous = check_variance_homogeneity(all_groups)
            if is_homogeneous:
                st.success(f"Variance Homogeneity Test: P-value = {p_variance:.4f} (Pass)")
            else:
                st.warning(f"Variance Homogeneity Test: P-value = {p_variance:.4f} (Fail)")
        except Exception as e:
            st.error(f"Error in Variance Homogeneity Test: {e}")

    parametric = all(check_normality(group)[1] for group in all_groups)
    if parametric:
        st.info("Your data is parametric.")
    else:
        st.info("Your data is non-parametric.")

    # Recommend the best test based on assumptions
    if parametric and len(all_groups) == 2:
        st.success("Recommended Test: Independent T-Test or Dependent T-Test (if groups are paired).")
    elif parametric and len(all_groups) > 2:
        st.success("Recommended Test: One-Way ANOVA.")
    elif not parametric and len(all_groups) == 2:
        st.success("Recommended Test: Mann-Whitney U Test or Wilcoxon Signed-Rank Test (if groups are paired).")
    elif not parametric and len(all_groups) > 2:
        st.success("Recommended Test: Kruskal-Wallis Test or Friedman Test (if groups are repeated measures).")
    else:
        st.info("Please ensure your data and assumptions match the test requirements.")

# Step 3: Hypothesis Testing
st.markdown("<h2 style='text-align: center;'>Step 3: Select and Perform a Hypothesis Test</h2>", unsafe_allow_html=True)

if data_type == "Numerical Data":
    test_list = numerical_tests_parametric if parametric else numerical_tests_nonparametric
else:
    test_list = categorical_tests

selected_test = st.selectbox("Choose the specific test to perform:", list(test_list.keys()))

# Display test description
st.info(f"**Test Description:** {test_list[selected_test]}")

if st.button("Run Test"):
    try:
        if selected_test == "Independent T-Test" and len(all_groups) >= 2:
            t_stat, p_value = stats.ttest_ind(all_groups[0], all_groups[1])
            st.success(f"Independent T-Test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        elif selected_test == "Dependent (Paired) T-Test" and len(all_groups) >= 2:
            t_stat, p_value = stats.ttest_rel(all_groups[0], all_groups[1])
            st.success(f"Dependent T-Test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        elif selected_test == "One-Way ANOVA" and len(all_groups) > 2:
            f_stat, p_value = stats.f_oneway(*all_groups)
            st.success(f"One-Way ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
        elif selected_test == "Mann-Whitney U Test" and len(all_groups) >= 2:
            u_stat, p_value = stats.mannwhitneyu(all_groups[0], all_groups[1])
            st.success(f"Mann-Whitney U Test: U-statistic = {u_stat:.4f}, p-value = {p_value:.4f}")
        else:
            st.error("The selected test is not implemented or requires more groups.")
    except Exception as e:
        st.error(f"An error occurred while performing the test: {e}")

