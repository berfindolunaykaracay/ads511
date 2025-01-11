import streamlit as st
import pandas as pd
from scipy.stats import shapiro, levene, ttest_rel, ttest_ind, mannwhitneyu, kruskal, wilcoxon, f_oneway
from scipy.stats import chi2_contingency, fisher_exact

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Application</h1>",
            unsafe_allow_html=True)

# Step 1: Veri Yükleme
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 1: Upload Dataset</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

# Kullanıcı bir dosya yüklediyse
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

    # Step 2: Sütun Seçimi ve Hipotezlerin Tanımlanması
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Column and Define Hypotheses</h2>",
                unsafe_allow_html=True)
    selected_columns = st.multiselect("Select columns for hypothesis testing", data.columns.tolist())

    if selected_columns:
        selected_test_col = st.selectbox("Select a column for hypothesis testing", selected_columns)

        # Hipotezleri Tanımlama
        st.markdown("<h3 style='text-align: center;'>Define Your Hypotheses</h3>", unsafe_allow_html=True)
        null_hypothesis = st.text_input("Enter the Null Hypothesis (H₀)", value="No significant difference.")
        alternative_hypothesis = st.text_input("Enter the Alternative Hypothesis (H₁)",
                                               value="There is a significant difference.")

        # Seçilen Sütunun Önizlemesi
        st.write(f"### Preview of Selected Column: {selected_test_col}")
        st.dataframe(data[[selected_test_col]].dropna().head())

        # Veri Türü Tespiti ve Assumption Check
        if pd.api.types.is_numeric_dtype(data[selected_test_col]):
            st.write(f"**Column '{selected_test_col}' is numeric.** Assumption checks will be performed.")
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 3: Assumption Check</h2>",
                        unsafe_allow_html=True)

            # Normality Check
            st.write("### a) Normality Test")
            normality_p_value = shapiro(data[selected_test_col].dropna())[1]
            st.write(f"Shapiro-Wilk Test P-Value: {normality_p_value}")
            normality_result = "Passes" if normality_p_value > 0.05 else "Fails"

            # Homogeneity of Variance Check
            st.write("### b) Homogeneity of Variance")
            if len(data[selected_test_col].unique()) > 1:
                levene_p_value = levene(data[selected_test_col].dropna(), data[selected_test_col].dropna())[1]
                st.write(f"Levene Test P-Value: {levene_p_value}")
                homogeneity_result = "Passes" if levene_p_value > 0.05 else "Fails"
            else:
                homogeneity_result = "Not Applicable"

            # Outlier Analysis
            st.write("### c) Outlier Analysis")
            z_scores = (data[selected_test_col] - data[selected_test_col].mean()) / data[selected_test_col].std()
            outliers = data[selected_test_col][(z_scores < -3) | (z_scores > 3)]
            st.write(f"Number of Outliers Detected: {len(outliers)}")

            # Parametrik mi Nonparametrik mi?
            if normality_result == "Passes" and homogeneity_result == "Passes":
                st.write("The assumptions are satisfied. You can proceed with parametric tests.")
                parametric = True
            else:
                st.write("The assumptions are not satisfied. Proceeding with non-parametric tests.")
                parametric = False

        elif pd.api.types.is_categorical_dtype(data[selected_test_col]) or data[selected_test_col].dtype == object:
            st.write(f"**Column '{selected_test_col}' is categorical.** Assumption checks are not required.")
            parametric = False

        # Step 4: Test Seçimi
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 4: Decide Test Type</h2>",
                    unsafe_allow_html=True)
        if parametric:
            test_options = ["Paired T-Test", "Independent T-Test", "One-Way ANOVA"]
        else:
            if pd.api.types.is_numeric_dtype(data[selected_test_col]):
                test_options = ["Mann-Whitney U Test", "Kruskal-Wallis Test", "Wilcoxon Signed-Rank Test"]
            else:
                test_options = ["Chi-Square Test", "Fisher Exact Test"]

        selected_test = st.radio("Select the test to perform", test_options)

        # Step 5: Testin Gerçekleştirilmesi
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 5: Run Hypothesis Test</h2>",
                    unsafe_allow_html=True)
        if st.button("Run Test"):
            try:
                if selected_test == "Paired T-Test":
                    t_stat, p_val = ttest_rel(data[selected_test_col].dropna()[:-1],
                                              data[selected_test_col].dropna()[1:])
                    st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")
                elif selected_test == "Mann-Whitney U Test":
                    u_stat, p_val = mannwhitneyu(data[selected_test_col].dropna()[:len(data) // 2],
                                                 data[selected_test_col].dropna()[len(data) // 2:])
                    st.write(f"U-Statistic: {u_stat}, P-Value: {p_val}")
                elif selected_test == "Chi-Square Test":
                    contingency_table = pd.crosstab(data[selected_test_col], data[selected_test_col])
                    chi2, p_val, _, _ = chi2_contingency(contingency_table)
                    st.write(f"Chi2 Statistic: {chi2}, P-Value: {p_val}")
                elif selected_test == "One-Way ANOVA":
                    col_data = data[selected_test_col].dropna()
                    groups = [col_data[:len(col_data) // 3], col_data[len(col_data) // 3:2 * len(col_data) // 3],
                              col_data[2 * len(col_data) // 3:]]
                    f_stat, p_val = f_oneway(*groups)
                    st.write(f"F-Statistic: {f_stat}, P-Value: {p_val}")
                # Diğer testler buraya eklenebilir
            except Exception as e:
                st.error(f"An error occurred while running the test: {e}")
else:
    st.info("Please upload a dataset to proceed.")
