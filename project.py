import streamlit as st
import pandas as pd

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing Application</h1>", unsafe_allow_html=True)

# Step 1: Veri Yükleme
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 1: Upload Dataset</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

# Kullanıcı bir dosya yüklediyse
if uploaded_file:
    # Veriyi yükle ve detaylı bilgileri göster
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center;'>Dataset Preview</h2>", unsafe_allow_html=True)
    st.write(f"**Number of Rows:** {data.shape[0]}")
    st.write(f"**Number of Columns:** {data.shape[1]}")
    st.write("### Column Information")
    column_details = pd.DataFrame({
        "Column Name": data.columns,
        "Data Type": data.dtypes,
        "Number of Unique Values": [data[col].nunique() for col in data.columns],
        "Number of Missing Values": [data[col].isnull().sum() for col in data.columns]
    })
    st.dataframe(column_details)

    st.write("### Dataset Head")
    st.dataframe(data.head())

    # Step 2: Sütun Seçimi
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Step 2: Select Column</h2>", unsafe_allow_html=True)
    selected_columns = st.multiselect("Select columns for hypothesis testing", data.columns.tolist())

    # Seçilen sütunları göster
    if selected_columns:
        st.markdown("<h3 style='text-align: center;'>Selected Columns</h3>", unsafe_allow_html=True)
        st.dataframe(data[selected_columns])
else:
    st.info("Please upload a dataset to proceed.")
