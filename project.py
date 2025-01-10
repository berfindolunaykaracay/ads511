import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    st.error("The required module 'seaborn' is not installed. Please install it by running 'pip install seaborn'.")

# Başlık
st.title("Data Input and Visualization App")
st.sidebar.header("Data Input")

# Veri girişi seçim
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
        if 'sns' in locals():
            if plot_type == "Histogram":
                sns.histplot(data[col], kde=True, ax=ax)
            elif plot_type == "Boxplot":
                sns.boxplot(x=data[col], ax=ax)
            elif plot_type == "Scatterplot":
                x_col = st.sidebar.selectbox("Select X-axis column", data.columns)
                y_col = st.sidebar.selectbox("Select Y-axis column", data.columns)
                sns.scatterplot(x=data[x_col], y=data[y_col], ax=ax)

            st.pyplot(fig)
        else:
            st.error("Seaborn is not available, so visualizations cannot be displayed.")

else:
    st.write("Please upload a CSV file or manually enter data to begin.")
