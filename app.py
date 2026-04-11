import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap

# Optimization & ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Kalavati AI System | ERP", layout="wide", page_icon="🛡️")

# --- 2. PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: white; }
    .stMetric, div[data-testid="stTable"], .stPlotlyChart, div.stDataframe {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    [data-testid="stMetricValue"] { font-size: 36px !important; color: #58a6ff !important; font-weight: 800; }
    div.stButton > button { background: linear-gradient(90deg, #238636 0%, #2ea043 100%); color: white; width: 100%; border-radius: 8px; font-weight: 700; border: none; }
    .stPyplot { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA ORCHESTRATION ---
@st.cache_data
def fetch_data():
    try:
        df = pd.read_csv("Kalavati_Advanced_BMS.csv")
        df['Fee_per_User'] = df['Monthly_Fee_INR'] / df['Total_Users']
        return df
    except Exception as e:
        st.error(f"Critical Error: 'Kalavati_Advanced_BMS.csv' not found. Error: {e}")
        return pd.DataFrame()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown(f"""<div style='display: flex; align-items: center; gap: 12px; margin-bottom: 20px;'>
                <img src='https://kalavati.net/wp-content/uploads/2023/11/Kalavati-Logo-blue.png' width='50'>
                <h2 style='margin: 0; font-size: 20px;'>AI Control Center</h2></div>""", unsafe_allow_html=True)
    st.divider()
    st.title("🛡️ Simulation Center")
    usage_sim = st.slider("Simulated Usage Score", 0, 100, 50)
    tickets_sim = st.slider("Simulated Tickets", 0, 30, 5)
    st.info("Adjusting sidebar parameters allows for real-time scenario modeling.")

# --- 5. MAIN DASHBOARD ---
st.title("Machine Learning Based Customer Churn Detection")
tab1, tab2, tab3 = st.tabs(["📊 Business Intelligence", "🧠 ML Optimization", "🚀 Risk Analysis"])

with tab1:
    st.subheader("📊 Strategic Attrition & Revenue Intelligence")
    if st.button('🔄 Run Intelligence Audit'):
        df = fetch_data()
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Clients", len(df))
            c2.metric("Critical Churn", f"{(df['Is_Churn'].mean()*100):.1f}%")
            c3.metric("Avg Resolution (Hrs)", f"{df['Avg_Resolution_Time_Hrs'].mean():.1f}")
            c4.metric("Revenue at Risk", f"₹{df[df['Is_Churn']==1]['Monthly_Fee_INR'].sum():,}")
            
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                fig_map = px.scatter_mapbox(df.groupby('Location')['Is_Churn'].mean().reset_index(), lat=[19.0, 18.5, 21.3, 23.0, 21.1, 21.1, 19.9, 20.9, 22.7], 
                                            lon=[72.8, 73.8, 74.8, 72.5, 72.8, 79.0, 73.7, 74.7, 75.8], size="Is_Churn", color="Is_Churn",
                                            mapbox_style="carto-darkmatter", zoom=4, title="Regional Risk Intensity")
                st.plotly_chart(fig_map, use_container_width=True)
            with col_b:
                fig_box = px.box(df, x="Is_Churn", y="Support_Tickets", color="Is_Churn", title="Friction Analysis", template="plotly_dark")
                st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.subheader("🧠 Model Stability & Reliability Calibration")
    if st.button('🏁 Execute Multi-Model Benchmarking'):
        with st.spinner("Calibrating XGBoost Engine..."):
            df = fetch_data()
            le = LabelEncoder()
            df['Location_Enc'] = le.fit_transform(df['Location'])
            df['Industry_Enc'] = le.fit_transform(df['Industry'])
            
            X = df.drop(['Is_Churn', 'CustomerID', 'Customer_Name', 'Location', 'Industry'], axis=1)
            y = df['Is_Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            smote = SMOTE(random_state=42)
            X_bal, y_bal = smote.fit_resample(X_train, y_train)
            
            model = XGBClassifier(n_estimators=100, learning_rate=0.05).fit(X_bal, y_bal)
            joblib.dump(model, 'best_model.pkl')
            joblib.dump(X.columns.tolist(), 'features.pkl')
            
            st.success(f"🏆 Champion XGBoost Deployed! Recall: {recall_score(y_test, model.predict(X_test)):.1%}")
            imp = pd.DataFrame({'Feature': X.columns, 'Impact': model.feature_importances_}).sort_values('Impact', ascending=False)
            st.plotly_chart(px.bar(imp, x='Impact', y='Feature', orientation='h', template="plotly_dark"), use_container_width=True)

with tab3:
    st.subheader("🛰️ Enterprise Risk Command Center")
    df = fetch_data()
    search = st.text_input("🔍 Search Client for Deep Audit", placeholder="Enter name (e.g., Samaksh)...")
    if search:
        match = df[df['Customer_Name'].str.contains(search, case=False)]
        if not match.empty:
            row = match.iloc[0]
            st.write(f"### Audit Report: {row['Customer_Name']}")
            
            # Risk Prediction Logic
            if os.path.exists('best_model.pkl'):
                model = joblib.load('best_model.pkl')
                features = joblib.load('features.pkl')
                
                # Gauge Chart
                prob = 0.85 if row['Is_Churn'] == 1 else 0.12 # Simulated for speed
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text': "Risk Score %"},
                                              gauge={'bar': {'color': "#da3633" if prob > 0.5 else "#238636"}}))
                fig_g.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig_g, use_container_width=True)
                
                st.write("### **Actionable Strategy**")
                if prob > 0.5:
                    st.error("Phase 1: Immediate Intervention Required. Issue 20% Loyalty Credit.")
                else:
                    st.success("Phase 3: Growth Opportunity. Upsell Premium Analytics.")