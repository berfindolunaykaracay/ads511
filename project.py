import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Başlık
st.title("Data Input and Visualization App")
st.sidebar.header("Data Input")

# Veri girişi seçimi
data_input_method = st.sidebar.radio("Select Data Input Method", ["Upload CSV", "Manual Entry"])

if data_input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("## Dataset Preview")
        st.write(data.head())

elif data_input_method == "Manual Entry":
    st.write("## Manual Data Entry")
    num_rows = st.number_input("Number of Rows", min_value=1, value=5)
    num_cols = st.number_input("Number of Columns", min_value=1, value=3)
    columns = [f"Column {i+1}" for i in range(num_cols)]

    manual_data = []
    for row in range(num_rows):
        row_data = []
        for col in columns:
            value = st.text_input(f"Row {row+1}, {col}", key=f"{row}-{col}")
            row_data.append(value)
        manual_data.append(row_data)

    data = pd.DataFrame(manual_data, columns=columns)
    st.write("## Dataset Preview")
    st.write(data)

# Veri gösterimi
if 'data' in locals():
    st.write("## Full Dataset Table")
    st.dataframe(data)

    # Görselleştirme seçenekleri
    st.sidebar.header("Visualization")
    if st.sidebar.checkbox("Show Data Visualization"):
        col = st.sidebar.selectbox("Select column to visualize", data.columns)
        plot_type = st.sidebar.selectbox("Select plot type", ["Histogram", "Boxplot", "Scatterplot"])

        fig, ax = plt.subplots()
        if plot_type == "Histogram":
            data[col].dropna().hist(ax=ax, bins=20, edgecolor="black")
            ax.set_title(f"Histogram of {col}")
        elif plot_type == "Boxplot":
            data[col].dropna().plot(kind='box', ax=ax)
            ax.set_title(f"Boxplot of {col}")
        elif plot_type == "Scatterplot":
            x_col = st.sidebar.selectbox("Select X-axis column", data.columns)
            y_col = st.sidebar.selectbox("Select Y-axis column", data.columns)
            ax.scatter(data[x_col], data[y_col], alpha=0.7)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatterplot of {x_col} vs {y_col}")

        st.pyplot(fig)
else:
    st.write("Please upload a CSV file or manually enter data to begin.")
