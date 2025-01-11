import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu, wilcoxon
import matplotlib.pyplot as plt

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Advanced Hypothesis Testing App</h1>", unsafe_allow_html=True)

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

        # Step 3: Veri Görselleştirme
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 3: Data Visualization</h2>", unsafe_allow_html=True)
        if len(selected_columns) == 1:
            st.write(f"### Distribution of {selected_columns[0]}")
            fig, ax = plt.subplots()
            data[selected_columns[0]].plot(kind='hist', ax=ax, bins=20, alpha=0.7)
            ax.set_title(f"Histogram of {selected_columns[0]}")
            st.pyplot(fig)
        elif len(selected_columns) == 2:
            st.write(f"### Relationship between {selected_columns[0]} and {selected_columns[1]}")
            fig, ax = plt.subplots()
            data.boxplot(column=selected_columns[0], by=selected_columns[1], ax=ax)
            plt.suptitle("")
            st.pyplot(fig)

        # Step 4: Hipotez testi için sütun seçimi
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 4: Select Columns for Hypothesis Testing</h2>", unsafe_allow_html=True)
        testing_columns = st.multiselect("Select columns for testing", selected_columns)

        if testing_columns:
            st.write(f"### Selected Columns for Testing: {', '.join(testing_columns)}")

            # Step 5: Test Önerileri ve Çoklu Seçim
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 5: Recommended Tests</h2>", unsafe_allow_html=True)
            recommendations = {}

            for col in testing_columns:
                rec_list = []
                if pd.api.types.is_numeric_dtype(data[col]):
                    unique_values = data[col].nunique()
                    if unique_values <= 10:
                        dependency = "Dependent" if data[col].duplicated().any() else "Independent"
                        if dependency == "Dependent":
                            rec_list.append("Wilcoxon Signed-Rank Test (dependent numerical groups)")
                        else:
                            rec_list.append("Mann-Whitney U Test (independent numerical groups)")
                    else:
                        rec_list.append("One-Way ANOVA (independent numerical groups)" if unique_values > 2 else "Independent T-Test")

                else:
                    unique_values = data[col].nunique()
                    if unique_values == 2:
                        dependency = "Dependent" if data[col].duplicated().any() else "Independent"
                        if dependency == "Dependent":
                            rec_list.append("McNemar Test (dependent categorical data)")
                        else:
                            rec_list.append("Chi-Square Test (independent categorical data)")
                    elif unique_values > 2:
                        rec_list.append("Chi-Square Test (independent categorical data with >2 categories)")

                recommendations[col] = rec_list

            for col, tests in recommendations.items():
                st.write(f"- **{col}:**")
                for test in tests:
                    st.write(f"  - {test}")

            # Step 6: Test Gerçekleştirme
            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 6: Perform Hypothesis Tests</h2>", unsafe_allow_html=True)
            for col, tests in recommendations.items():
                st.write(f"### Performing Tests for {col}")
                for test in tests:
                    if st.button(f"Run {test} for {col}"):
                        try:
                            if test == "Wilcoxon Signed-Rank Test (dependent numerical groups)":
                                col_data = data[col].dropna()
                                stat, p_val = wilcoxon(col_data[:-1], col_data[1:])
                                st.write(f"Statistic: {stat}, P-Value: {p_val}")
                            elif test == "Mann-Whitney U Test (independent numerical groups)":
                                col_data = data[col].dropna()
                                group1 = col_data[:len(col_data)//2]
                                group2 = col_data[len(col_data)//2:]
                                stat, p_val = mannwhitneyu(group1, group2)
                                st.write(f"Statistic: {stat}, P-Value: {p_val}")
                            elif test == "One-Way ANOVA (independent numerical groups)":
                                col_data = data[col].dropna()
                                groups = [col_data[:len(col_data)//3], col_data[len(col_data)//3:2*len(col_data)//3], col_data[2*len(col_data)//3:]]
                                f_stat, p_val = f_oneway(*groups)
                                st.write(f"F-Statistic: {f_stat}, P-Value: {p_val}")
                            elif test == "Chi-Square Test (independent categorical data)":
                                contingency_table = pd.crosstab(data[col], data[col])
                                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                                st.write(f"Chi2 Statistic: {chi2}, P-Value: {p_val}")
                        except Exception as e:
                            st.write(f"Error while performing the test: {e}")
        else:
            st.write("No columns selected for testing.")
    else:
        st.write("No columns selected.")
