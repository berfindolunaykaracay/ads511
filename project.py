import streamlit as st
import pandas as pd

# Başlık
st.title("Hypothesis Testing App")

# CSV dosyası yükleme
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("## Dataset Preview")
    st.write(data.head())
