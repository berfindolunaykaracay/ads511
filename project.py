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
st.sidebar.image("TEDU_LOGO.png", use_container_width=True)
st.sidebar.title("ADS 511: Statistical Inference Methods")
st.sidebar.write("Developed by: Serdar Hosver")

st.sidebar.title("Hypothesis Testing Map")
st.sidebar.image("Hypothesis_Test_Map.png", use_container_width=True)

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
st.title("Hypothesis Testing Application")
st.markdown("This app allows you to perform various **hypothesis tests** with ease. Simply upload your data or enter it manually, and let the app guide you through hypothesis testing.")

# Step 1: Data Input
st.header("Step 1: Data Input")
data_choice = st.radio("How would you like to input your data?", ("Upload CSV", "Enter Manually"))

all_groups = []
if data_choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        columns = st.multiselect("Select columns for testing", options=data.columns)
        if columns:
            all_groups = [data[col].dropna().tolist() for col in columns]

elif data_choice == "Enter Manually":
    st.write("Enter your data as arrays. Type each array separately.")
    group_input = st.text_area("Enter arrays (e.g., [1,2,3]) one by one, separated by new lines.")
    if group_input:
        try:
            all_groups = [eval(line.strip()) for line in group_input.splitlines() if line.strip()]
        except Exception as e:
            st.error(f"Error in parsing input: {e}")

if not all_groups:
    st.warning("No valid data provided.")
    st.stop()

# Step 1.5: Data Type Selection
data_type = st.radio(
    "What is your data type?",
    options=["Select", "Numerical Data", "Categorical Data"],
    index=0
)

if data_type == "Select":
    st.warning("Please select your data type to proceed.")
    st.stop()

# Step 2: Assumption Checks (if Numerical Data)
if data_type == "Numerical Data":
    st.header("Step 2: Assumption Check")
    st.write("Performing Normality and Variance Homogeneity Tests")

    results = []
    for i, group in enumerate(all_groups, start=1):
        try:
            p_normality, is_normal = check_normality(group)
            results.append((f"Group {i}", "Normality", p_normality, "Pass" if is_normal else "Fail"))
        except ValueError as e:
            st.error(f"Error with Group {i}: {e}")

    if len(all_groups) > 1:
        try:
            p_variance, is_homogeneous = check_variance_homogeneity(all_groups)
            results.append(("All Groups", "Variance Homogeneity", p_variance, "Pass" if is_homogeneous else "Fail"))
        except Exception as e:
            st.error(f"Error in Variance Homogeneity Test: {e}")

    results_df = pd.DataFrame(results, columns=["Group", "Test", "P-value", "Result"])
    st.table(results_df)

    parametric = all(res[3] == "Pass" for res in results)
    if parametric:
        st.info("Your data is parametric.")
    else:
        st.info("Your data is non-parametric.")

# Step 3: Hypothesis Testing
st.header("Step 3: Select and Perform a Hypothesis Test")

if data_type == "Numerical Data":
    test_list = numerical_tests_parametric if parametric else numerical_tests_nonparametric
else:
    test_list = categorical_tests

selected_test = st.selectbox("Choose the specific test to perform:", list(test_list.keys()))

# Display test description
st.info(f"**Test Description:** {test_list[selected_test]}")

if st.button("Run Test"):
    st.write(f"Performing {selected_test}...")
    # Add test-specific implementation logic here.

st.write("Thank you for using the Hypothesis Testing Application!")
