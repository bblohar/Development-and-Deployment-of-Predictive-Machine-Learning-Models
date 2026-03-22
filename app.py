import streamlit as st
import pandas as pd
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt

# Professional Page Config
st.set_page_config(page_title="Kalavati Tech ML Dashboard", layout="wide")

st.title("🚀 Predictive ML Model: Development & Deployment")
st.write("### Month 2 & 3: Data Ingestion & Exploratory Data Analysis")

# --- DATABASE CONNECTION ---
def get_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Spyrob@2909", # Use your password here
        database="kalavati_db"
    )
    df = pd.read_sql("SELECT * FROM customer_churn_data", conn)
    conn.close()
    return df

# --- UI LAYOUT ---
if st.button('🔄 Fetch Real-Time Data from SQL'):
    try:
        df = get_data()
        st.success("Data Ingestion Layer: Successful!")
        
        # Display Raw Data (Ingestion Layer)
        st.subheader("1. Data Ingestion Layer (Raw DataFrames)")
        st.dataframe(df.head(10)) 

        # Display EDA (Intelligence Phase)
        st.subheader("2. Exploratory Data Analysis (EDA)")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Churn Distribution**")
            fig, ax = plt.subplots()
            sns.countplot(x='Is_Churn', data=df, ax=ax, palette='viridis')
            st.pyplot(fig)

        with col2:
            st.write("**Feature Correlation Heatmap**")
            fig, ax = plt.subplots()
            # Selecting only numeric columns for correlation
            numeric_df = df.select_dtypes(include=['number'])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Connection Failed: {e}")