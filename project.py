import streamlit as st
import pandas as pd
from scipy.stats import f_oneway, ttest_ind

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
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Columns for Testing</h2>", unsafe_allow_html=True)
    categorical_column = st.selectbox("Select a categorical column (e.g., parent level of education)", data.columns.tolist())
    numerical_column = st.selectbox("Select a numerical column (e.g., math score)", data.columns.tolist())

    if categorical_column and numerical_column:
        st.markdown(f"<h3 style='text-align: center;'>Selected Columns: {categorical_column} (Categorical), {numerical_column} (Numerical)</h3>", unsafe_allow_html=True)

        # Veri Kontrolü
        if pd.api.types.is_numeric_dtype(data[numerical_column]) and not pd.api.types.is_numeric_dtype(data[categorical_column]):
            unique_groups = data[categorical_column].nunique()
            st.write(f"**Number of groups in {categorical_column}:** {unique_groups}")

            if unique_groups > 1:
                # Test Seçimi
                if unique_groups == 2:
                    st.markdown("<h4 style='text-align: center;'>Performing Independent T-Test</h4>", unsafe_allow_html=True)
                    groups = data[categorical_column].unique()
                    group1 = data[data[categorical_column] == groups[0]][numerical_column].dropna()
                    group2 = data[data[categorical_column] == groups[1]][numerical_column].dropna()

                    if len(group1) > 1 and len(group2) > 1:
                        t_stat, p_val = ttest_ind(group1, group2)
                        st.success(f"T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4f}")
                        if p_val < 0.05:
                            st.write("Result: There is a significant effect of the categorical variable on the numerical variable.")
                        else:
                            st.write("Result: No significant effect of the categorical variable on the numerical variable.")
                    else:
                        st.error("Not enough data in one or both groups for Independent T-Test.")
                else:
                    st.markdown("<h4 style='text-align: center;'>Performing One-Way ANOVA</h4>", unsafe_allow_html=True)
                    groups = [data[data[categorical_column] == group][numerical_column].dropna() for group in data[categorical_column].unique()]

                    if all(len(group) > 1 for group in groups):
                        f_stat, p_val = f_oneway(*groups)
                        st.success(f"F-Statistic: {f_stat:.4f}, P-Value: {p_val:.4f}")
                        if p_val < 0.05:
                            st.write("Result: There is a significant effect of the categorical variable on the numerical variable.")
                        else:
                            st.write("Result: No significant effect of the categorical variable on the numerical variable.")
                    else:
                        st.error("Not enough data in one or more groups for One-Way ANOVA.")
            else:
                st.error("The selected categorical column must have at least two groups.")
        else:
            st.error("Please select a categorical column and a numerical column correctly.")
