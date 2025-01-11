import streamlit as st
import pandas as pd

# Sayfa stil ayarları
st.markdown(
    """
    <style>
        body {
            background-color: white;
            color: #002366;
        }
        h1 {
            color: #002366;
            text-align: center;
        }
        .stButton button {
            background-color: #002366;
            color: white;
            border-radius: 5px;
            font-size: 16px;
        }
        .stFileUploader {
            margin-top: 30px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Başlık
st.markdown("<h1>Hypothesis Testing App</h1>", unsafe_allow_html=True)

# Boşluk ve stil ayarları
st.markdown("<div style='margin-top: 30px; text-align: center; font-size: 20px;'>Upload a CSV file</div>", unsafe_allow_html=True)

# CSV dosyası yükleme
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("## Dataset Preview")
    st.write(data.head())
