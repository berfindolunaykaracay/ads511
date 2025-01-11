import streamlit as st
import pandas as pd

# Başlık
st.title("Hypothesis Testing App")

# Boşluk ve stil ayarları
st.write("\n\n")
st.header("Upload a CSV file")

# CSV dosyası yükleme
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("## Dataset Preview")
    st.write(data.head())
