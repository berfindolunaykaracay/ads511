import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
try:
    from statsmodels.stats.contingency_tables import mcnemar
except ModuleNotFoundError:
    st.error("Required module 'statsmodels' is not installed. Install it using 'pip install statsmodels'.")
    st.stop()
from scipy.stats import chi2_contingency, fisher_exact

# Configure pandas display options
pd.options.display.float_format = '{:,.4f}'.format

# Streamlit app configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Advanced Hypothesis Testing App",
    page_icon="📊",
    layout="wide",
)

# Set custom background style
page_bg = """
<style>
body {
    background-color: #f0f8ff;
    color: #333333;
    font-family: 'Arial', sans-serif;
}
.sidebar .sidebar-content {
    background-color: #e6e6fa;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Helper functions
def check_normality(data):
    """Check for normality using the Shapiro-Wilk test."""
    data = np.array(data)
    test_stat, p_value = stats.shapiro(data)
    return p_value, p_value >= 0.05

def check_variance_homogeneity(data):
    """Check homogeneity of variances using Levene's test."""
    if len(data) < 2:
        return None, False
    test_stat, p_value = stats.levene(*data)
    return p_value, p_value >= 0.05

# Hypothesis test descriptions
test_descriptions = {
    "t_test_independent": "Compare the means of two independent groups using the Independent T-Test.",
    "dependent_ttest": "Compare means of two related groups using the Dependent (Paired) T-Test.",
    "repeated_measure_anova": "Assess differences across multiple time points using Repeated Measures ANOVA.",
    "oneway_anova": "Test differences between three or more groups with One-Way ANOVA.",
    "Wilcoxon_signed_rank": "Non-parametric test for paired data: the Wilcoxon Signed-Rank Test.",
    "Mann_Whitney_U_Test": "Non-parametric test comparing two independent groups.",
    "Friedman_Chi_Square": "Non-parametric alternative to Repeated Measures ANOVA.",
    "Kruskal_Wallis": "Non-parametric alternative to One-Way ANOVA.",
    "McNemar_test": "Evaluate changes in categorical data using the McNemar Test.",
    "Chi_squared_test": "Test associations between categorical variables using the Chi-Squared Test.",
    "Fisher_exact_test": "Assess associations for small sample categorical data with Fisher's Exact Test.",
}

# Sidebar setup
st.sidebar.title("Statistical Testing Suite")
st.sidebar.header("Available Hypothesis Tests")
for test, description in test_descriptions.items():
    with st.sidebar.expander(f"ℹ️ {test}"):
        st.write(description)

# Main title and introduction
st.title("Welcome to the Hypothesis Testing App")
st.markdown(
    "Effortlessly perform hypothesis tests with a user-friendly interface. Upload your data or input manually to get started!"
)

# Step 1: Data input
st.header("Step 1: Input Your Data")
data_choice = st.selectbox(
    "How would you like to input your data?",
    ("Upload CSV/Excel File", "Enter Data Manually")
)

all_groups = []
if data_choice == "Upload CSV/Excel File":
    uploaded_file = st.file_uploader("Upload a file (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        file_type = "csv" if uploaded_file.name.endswith(".csv") else "excel"
        try:
            data = pd.read_csv(uploaded_file) if file_type == "csv" else pd.read_excel(uploaded_file)
            st.subheader("Preview of Uploaded Data:")
            st.dataframe(data.head(), use_container_width=True)
            columns = st.multiselect("Select columns to analyze:", data.columns)
            if columns:
                all_groups = [data[col].dropna().tolist() for col in columns]
        except Exception as e:
            st.error(f"Error reading file: {e}")
elif data_choice == "Enter Data Manually":
    st.write("Enter your data below. Each group should be on a new line, formatted as a Python list, e.g., [1, 2, 3].")
    group_input = st.text_area("Manual Data Input")
    if group_input:
        try:
            all_groups = [eval(line.strip()) for line in group_input.splitlines() if line.strip()]
        except Exception as e:
            st.error(f"Error processing input: {e}")

if not all_groups:
    st.warning("No valid data provided. Please upload or enter your data.")
    st.stop()

# Step 2: Assumption checks
st.header("Step 2: Assumption Checks")
st.markdown("We will evaluate normality and variance homogeneity where applicable.")

results = []
for i, group in enumerate(all_groups, start=1):
    p_normality, is_normal = check_normality(group)
    results.append((f"Group {i}", "Normality", p_normality, "Pass" if is_normal else "Fail"))

if len(all_groups) > 1:
    p_variance, is_homogeneous = check_variance_homogeneity(all_groups)
    results.append(("All Groups", "Variance Homogeneity", p_variance, "Pass" if is_homogeneous else "Fail"))

results_df = pd.DataFrame(results, columns=["Group", "Test", "P-value", "Result"])
st.table(results_df)

parametric = all(res[3] == "Pass" for res in results)
st.info("The data meets parametric assumptions." if parametric else "The data does not meet parametric assumptions.")

# Step 3: Hypothesis test selection
st.header("Step 3: Select and Perform a Hypothesis Test")
data_types = {
    "Numerical": ["t_test_independent", "dependent_ttest", "oneway_anova", "Wilcoxon_signed_rank"],
    "Categorical": ["McNemar_test", "Chi_squared_test", "Fisher_exact_test"],
}
data_type = st.radio("Choose the type of data you are working with", list(data_types.keys()))
selected_test = st.selectbox("Select a hypothesis test", data_types[data_type])

# Step 4: Run the selected hypothesis test
if st.button("Run Test"):
    try:
        if selected_test == "t_test_independent" and len(all_groups) >= 2:
            t_stat, p_value = stats.ttest_ind(all_groups[0], all_groups[1])
            st.write(f"Independent T-Test Results: t-stat = {t_stat:.4f}, p = {p_value:.4f}")
        elif selected_test == "Chi_squared_test" and len(all_groups) >= 2:
            contingency = pd.crosstab(all_groups[0], all_groups[1])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            st.write(f"Chi-Squared Test Results: chi2 = {chi2:.4f}, p = {p_value:.4f}")
        else:
            st.warning("The selected test requires a different data configuration.")
    except Exception as e:
        st.error(f"An error occurred while running the test: {e}")
