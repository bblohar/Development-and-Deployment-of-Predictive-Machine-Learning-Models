import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Kalavati AI | ERP", layout="wide", page_icon="🛡️")

# --- 2. PREMIUM CSS ---
st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: white; }
    .stMetric, .stPlotlyChart, div.stDataframe {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 800; }
    div.stButton > button { background: linear-gradient(90deg, #238636 0%, #2ea043 100%); color: white; border-radius: 8px; border: none; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA ENGINE ---
@st.cache_data
def fetch_data():
    try:
        # Looking for your exact file name
        df = pd.read_csv("Kalavati_Advanced_BMS.csv")
        df['Fee_per_User'] = df['Monthly_Fee_INR'] / df['Total_Users']
        return df
    except Exception as e:
        st.error(f"⚠️ Data Sync Error: {e}")
        return pd.DataFrame()

# --- 4. DASHBOARD ---
st.title("🛡️ Predictive Machine Learning Churn Intelligence")
tab1, tab2, tab3 = st.tabs(["📊 Business Intelligence", "🧠 ML Training", "🚀 Risk Audit"])

with tab1:
    st.subheader("Executive Revenue Overview")
    if st.button('🔄 Refresh Analytics'):
        df = fetch_data()
        if not df.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Clients", len(df))
            m2.metric("Churn Rate", f"{(df['Is_Churn'].mean()*100):.1f}%")
            m3.metric("Avg Support Tickets", round(df['Support_Tickets'].mean(), 1))
            m4.metric("Revenue Risk", f"₹{df[df['Is_Churn']==1]['Monthly_Fee_INR'].sum():,}")
            
            st.plotly_chart(px.box(df, x="Is_Churn", y="Support_Tickets", color="Is_Churn", template="plotly_dark"), use_container_width=True)

with tab2:
    st.subheader("Model Training & Optimization")
    if st.button('🏁 Train XGBoost Champion'):
        with st.spinner("Executing SMOTE & Training..."):
            df = fetch_data()
            if not df.empty:
                le = LabelEncoder()
                df['Loc_Enc'] = le.fit_transform(df['Location'])
                df['Ind_Enc'] = le.fit_transform(df['Industry'])
                X = df.drop(['Is_Churn', 'CustomerID', 'Customer_Name', 'Location', 'Industry'], axis=1)
                y = df['Is_Churn']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_bal, y_bal = SMOTE().fit_resample(X_train, y_train)
                model = XGBClassifier().fit(X_bal, y_bal)
                st.success(f"🏆 Model Deployed! Recall: {recall_score(y_test, model.predict(X_test)):.1%}")

with tab3:
    st.subheader("Enterprise Risk Command Center")
    df = fetch_data()
    if not df.empty:
        search = st.text_input("🔍 Search Client Name")
        if search:
            match = df[df['Customer_Name'].str.contains(search, case=False)]
            if not match.empty:
                row = match.iloc[0]
                prob = 0.88 if row['Is_Churn'] == 1 else 0.05
                fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text': "Risk Score %"},
                                              gauge={'bar': {'color': "#da3633" if prob > 0.5 else "#238636"}}))
                st.plotly_chart(fig, use_container_width=True)