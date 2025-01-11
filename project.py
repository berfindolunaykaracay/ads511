import streamlit as st
import pandas as pd

# Başlık
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hypothesis Testing App</h1>", unsafe_allow_html=True)

# Boşluk ve stil ayarları
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Upload a CSV file</h2>", unsafe_allow_html=True)

# CSV dosyası yükleme
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Dataset Preview</h2>", unsafe_allow_html=True)

    # Dataset boyutu
    st.write(f"**Number of rows:** {data.shape[0]}")
    st.write(f"**Number of columns:** {data.shape[1]}")

    # Tüm veri önizleme
    st.write("### **Full Dataset**")
    st.dataframe(data)

    # İlk 9 sütun için birden fazla seçim yapma
    st.write("### **Select Columns from First 9 Columns**")
    first_9_columns = data.columns[:9]
    selected_columns = st.multiselect("Select columns", first_9_columns)

    if selected_columns:
        st.write(f"### **Data for Selected Columns:** {', '.join([f'*{col}*' for col in selected_columns])}")
        st.dataframe(data[selected_columns].reset_index(drop=True))
    else:
        st.write("No columns selected.")

    # Teste tabi tutulacak sütunları seçme
    st.write("### **Select Columns for Hypothesis Testing**")
    num_columns = len(data.columns)
    rows = num_columns // 3 + (num_columns % 3 > 0)
    testing_columns = []

    for row in range(rows):
        cols = st.columns(3)
        for i in range(3):
            idx = row * 3 + i
            if idx < num_columns:
                col_name = data.columns[idx]
                if cols[i].checkbox(col_name, key=f"test_col_{col_name}"):
                    testing_columns.append(col_name)

    if testing_columns:
        st.write(f"### **Selected Columns for Testing:** {', '.join([f'*{col}*' for col in testing_columns])}")
    else:
        st.write("No columns selected for testing.")
