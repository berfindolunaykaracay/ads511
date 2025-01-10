import streamlit as st
import pandas as pd

# Başlık
st.title("Hypothesis Testing App")

# Boşluk ve stil ayarları
st.markdown("<div style='margin-top: 30px; text-align: center; font-size: 20px;'>Upload a CSV file</div>", unsafe_allow_html=True)

# CSV dosyası yükleme
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("## Dataset Preview")
    st.write(data.head())
