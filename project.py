import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

    # Kategoriye göre veri gösterimi
    st.write("### Data by Categories")
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    if not categorical_columns.empty:
        for category_column in categorical_columns:
            st.write(f"#### Category: {category_column}")
            unique_values = data[category_column].dropna().unique()
            st.write(f"Unique Values: {', '.join(map(str, unique_values))}")
            st.dataframe(data[[category_column]].drop_duplicates().reset_index(drop=True))
    else:
        st.write("No categorical columns available.")

    # Görselleştirme seçenekleri
    st.markdown("<h2 style='text-align: center;'>Data Visualization</h2>", unsafe_allow_html=True)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    if not numeric_columns.empty:
        column_to_plot = st.selectbox("Select a column to visualize", numeric_columns)
        plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot"])

        fig, ax = plt.subplots()

        if plot_type == "Histogram":
            data[column_to_plot].plot(kind='hist', bins=20, edgecolor='black', ax=ax)
            ax.set_title(f"Histogram of {column_to_plot}")
            ax.set_xlabel(column_to_plot)
            ax.set_ylabel("Frequency")
        elif plot_type == "Boxplot":
            data[column_to_plot].plot(kind='box', ax=ax)
            ax.set_title(f"Boxplot of {column_to_plot}")

        st.pyplot(fig)
    else:
        st.write("No numeric columns available for visualization.")
