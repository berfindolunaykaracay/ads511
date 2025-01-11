import streamlit as st
import pandas as pd

# Başlık
st.markdown("<h1 style='text-align: center;'>Hypothesis Testing App</h1>", unsafe_allow_html=True)

# Boşluk ve stil ayarları
st.markdown("<h2 style='text-align: center;'>Upload a CSV file</h2>", unsafe_allow_html=True)

# CSV dosyası yükleme
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center;'>Dataset Preview</h2>", unsafe_allow_html=True)

    # Dataset boyutu
    st.write(f"**Number of rows:** {data.shape[0]}")
    st.write(f"**Number of columns:** {data.shape[1]}")

    # Tüm veri önizleme
    st.write("### Full Dataset")
    st.dataframe(data)

    # İlk 9 sütun için birden fazla seçim yapma
    st.write("### Select Columns from First 9 Columns")
    first_9_columns = data.columns[:9]
    selected_columns = st.multiselect("Select columns", first_9_columns)

    if selected_columns:
        st.write(f"### Data for Selected Columns: {', '.join(selected_columns)}")
        st.dataframe(data[selected_columns].reset_index(drop=True))
    else:
        st.write("No columns selected.")

    # Teste tabi tutulacak sütunları seçme
    st.write("### Select Columns for Hypothesis Testing")
    test_columns = st.columns(len(data.columns))
    testing_columns = []

    for i, col in enumerate(data.columns):
        if test_columns[i].checkbox(col, key=f"test_col_{col}"):
            testing_columns.append(col)

    if testing_columns:
        st.write(f"### Selected Columns for Testing: {', '.join(testing_columns)}")
    else:
        st.write("No columns selected for testing.")
