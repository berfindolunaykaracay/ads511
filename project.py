import streamlit as st
import pandas as pd

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Guide</h1>", unsafe_allow_html=True)

# CSV Dosyası Yükleme
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Upload a CSV file</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Dataset Preview</h2>", unsafe_allow_html=True)

    # Dataset bilgisi
    st.write(f"**Number of rows:** {data.shape[0]}")
    st.write(f"**Number of columns:** {data.shape[1]}")
    st.write("### Full Dataset")
    st.dataframe(data)

    # İlk 9 sütun seçimi
    st.write("### Select Columns")
    first_9_columns = data.columns[:9]
    selected_columns = st.multiselect("Select columns", first_9_columns)

    if selected_columns:
        st.write("### Data for Selected Columns")
        st.dataframe(data[selected_columns].reset_index(drop=True))

        # Hipotez testi için sütun seçimi
        st.write("### Select Columns for Hypothesis Testing")
        testing_columns = []

        for col in selected_columns:
            if st.checkbox(f"Include {col}", key=f"test_col_{col}"):
                testing_columns.append(col)

        if testing_columns:
            st.write(f"### Selected Columns for Testing: {', '.join(testing_columns)}")

            # Veri türü otomatik tespiti ve uygun testlerin önerilmesi
            st.write("### Recommended Tests")
            recommendations = []

            for col in testing_columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Sayısal veri için test önerileri
                    group_count = st.radio(f"How many groups does {col} have?", ["2 Groups", ">2 Groups"], key=f"group_{col}")
                    dependency = st.radio(f"Are the groups dependent or independent?", ["Dependent", "Independent"], key=f"dep_{col}")

                    if group_count == "2 Groups":
                        if dependency == "Dependent":
                            recommendations.append((col, "Paired T-Test (tests dependent numerical groups)"))
                        else:
                            recommendations.append((col, "Independent T-Test (tests independent numerical groups)"))
                    else:
                        if dependency == "Dependent":
                            recommendations.append((col, "Repeated Measures ANOVA (tests dependent numerical groups)"))
                        else:
                            recommendations.append((col, "One-Way ANOVA (tests independent numerical groups)"))

                else:
                    # Kategorik veri için test önerileri
                    group_count = st.radio(f"How many categories does {col} have?", ["2 Categories", ">2 Categories"], key=f"cat_group_{col}")
                    dependency = st.radio(f"Are the categories dependent or independent?", ["Dependent", "Independent"], key=f"cat_dep_{col}")

                    if group_count == "2 Categories":
                        if dependency == "Dependent":
                            recommendations.append((col, "McNemar Test (tests dependent categorical data)"))
                        else:
                            recommendations.append((col, "Chi-Square Test (tests independent categorical data)"))
                    else:
                        if dependency == "Dependent":
                            recommendations.append((col, "Marginal Homogeneity Test (tests dependent categorical data with >2 categories)"))
                        else:
                            recommendations.append((col, "Chi-Square Test (tests independent categorical data with >2 categories)"))

            if recommendations:
                for col, test in recommendations:
                    st.write(f"- **{col}:** {test}")
            else:
                st.write("No suitable tests found for the selected configuration.")
        else:
            st.write("No columns selected for testing.")
    else:
        st.write("No columns selected.")
