import mysql.connector
import pandas as pd
import streamlit as st

# --- DATABASE CONFIGURATION ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Spyrob@2909", # Update with your actual SQL Workbench password
    "database": "kalavati_db"
}

def get_live_data():
    """Fetches the full dataset for BI and ML tabs."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql("SELECT * FROM kalavati_advanced_bms_data", conn)
        conn.close()
        df['Fee_per_User'] = df['Monthly_Fee_INR'] / df['Total_Users']
        return df, "Live SQL Workbench"
    except:
        # Fallback to CSV if SQL connection is unavailable
        df = pd.read_csv("Kalavati_Advanced_BMS.csv")
        df['Fee_per_User'] = df['Monthly_Fee_INR'] / df['Total_Users']
        return df, "Production Snapshot (CSV)"

def search_sql_data(query, filter_col):
    """Handles the search engine in Tab 3."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        sql_query = f"SELECT * FROM kalavati_advanced_bms_data WHERE {filter_col} LIKE '%{query}%'"
        res_df = pd.read_sql(sql_query, conn)
        conn.close()
        return res_df
    except:
        return pd.DataFrame()

def fetch_sql_row(client_id):
    """Fetches a single client record for the Audit."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        sql_query = f"SELECT * FROM kalavati_advanced_bms_data WHERE CustomerID = '{client_id}'"
        row = pd.read_sql(sql_query, conn)
        conn.close()
        return row
    except:
        return pd.DataFrame()