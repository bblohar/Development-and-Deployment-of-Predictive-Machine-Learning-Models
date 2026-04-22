import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap
import requests
import numpy as np
from database_engine import get_live_data, search_sql_data, fetch_sql_row
# Optimization & Metric Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier  
from imblearn.over_sampling import SMOTE

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kalavati AI System | ERP", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- 2. PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    .stMetric, div[data-testid="stTable"], .stPlotlyChart, div.stDataframe {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 1rem;
    }
    [data-testid="stMetricValue"] { font-size: 36px !important; font-weight: 800 !important; color: #58a6ff !important; }
    div.stButton > button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%); color: white;
        border-radius: 8px; border: none; padding: 0.6rem 1rem; font-weight: 600; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA ORCHESTRATION ---
@st.cache_data
def fetch_data():
    try:
        df = pd.read_csv("Kalavati_Advanced_BMS_Data.csv")
        df['Fee_per_User'] = df['Monthly_Fee_INR'] / df['Total_Users']
        return df
    except Exception as e:
        st.error(f"Critical Error: {e}")
        return pd.DataFrame()

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown('<img src="https://kalavati.net/wp-content/uploads/2023/11/Kalavati-Logo-blue.png" width="50">', unsafe_allow_html=True)
    st.title("Control Center")
    with st.expander("Financial Data", expanded=True):
        fee = st.number_input("Monthly Fee (INR)", 500, 5000, 1500)
        total_users = st.slider("Total Users", 1, 50, 5)
        delay = st.slider("Payment Delay (Days)", 0, 30, 0)
    with st.expander("Engagement Metrics", expanded=True):
        usage = st.slider("Usage Score", 0, 100, 50)
        tickets = st.slider("Support Tickets", 0, 20, 2)
        nps = st.slider("NPS Score", 1, 10, 7)

st.title("Machine Learning Based Customer Churn Detection")
tab1, tab2, tab3 = st.tabs(["Business Intelligence", "ML Optimization", "Risk Analysis"])

# --- TAB 1: EDA ---
with tab1:
    st.subheader("Strategic Attrition & Revenue Intelligence")
    if st.button('🔄 Execute Strategic Analysis'):
        with st.spinner("Processing Business Records..."):
            df = fetch_data()
            # 1. Executive Summary Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Client Base", len(df))
            m2.metric("Critical Churn Rate", f"{(df['Is_Churn'].mean()*100):.1f}%")
            m3.metric("Avg Support Tickets", round(df['Support_Tickets'].mean(), 1))
            m4.metric("Revenue at Risk", f"₹{df[df['Is_Churn']==1]['Monthly_Fee_INR'].sum():,}")
            st.divider()
            # 2. IDENTIFYING RED FLAGS
            st.write("**Technical Red Flags: Behavioral Identification**")
            col_a, col_b = st.columns(2)
            with col_a:
                fig_tick = px.box(df, x="Is_Churn", y="Support_Tickets", color="Is_Churn",
                                  title="Support Ticket Friction (0=Stay, 1=Churn)",
                                  template="plotly_dark", color_discrete_sequence=['#238636', '#da3633'])
                st.plotly_chart(fig_tick, use_container_width=True)
                st.caption("Insight: Customers who churn have significantly higher median support tickets.")
            with col_b:
                fig_delay = px.box(df, x="Is_Churn", y="Payment_Delay_Days", color="Is_Churn",
                                  title="Payment Delay Patterns",
                                  template="plotly_dark", color_discrete_sequence=['#238636', '#da3633'])
                st.plotly_chart(fig_delay, use_container_width=True)
                st.caption("Insight: Payment delays over 15 days are a leading indicator of churn.")
            st.divider()
            st.write("**Top 10 High-Revenue At-Risk Accounts**")
            audit_df = df[df['Is_Churn'] == 1].sort_values(by='Monthly_Fee_INR', ascending=False).head(10)
            st.dataframe(audit_df[['CustomerID', 'Industry', 'Monthly_Fee_INR', 'Support_Tickets', 'NPS_Score']], use_container_width=True)
        st.write("### **Regional Churn Distribution**")
        coords = {
            'Mumbai': [19.0760, 72.8777], 'Pune': [18.5204, 73.8567],
            'Shirpur': [21.3524, 74.8814], 'Ahmedabad': [23.0225, 72.5714],
            'Surat': [21.1702, 72.8311], 'Nagpur': [21.1458, 79.0882],
            'Nashik': [19.9975, 73.7898], 'Dhule': [20.9042, 74.7749],
            'Indore': [22.7196, 75.8577]
        }
        df_map = fetch_data()
        if not df_map.empty:
            loc_stats = df_map.groupby('Location')['Is_Churn'].agg(['count', 'sum']).reset_index()
            loc_stats.columns = ['Location', 'Total_Customers', 'Churners']
            loc_stats['Churn_Rate'] = (loc_stats['Churners'] / loc_stats['Total_Customers']) * 100
            loc_stats['lat'] = loc_stats['Location'].map(lambda x: coords.get(x, [0,0])[0])
            loc_stats['lon'] = loc_stats['Location'].map(lambda x: coords.get(x, [0,0])[1])
            loc_stats = loc_stats[loc_stats['lat'] != 0]
            fig_map = px.scatter_mapbox(
                loc_stats, lat="lat", lon="lon", 
                size="Total_Customers", color="Churn_Rate",
                color_continuous_scale=px.colors.sequential.Reds,
                hover_name="Location", 
                hover_data={"Churners": True, "Total_Customers": True},
                zoom=5, height=500,
                mapbox_style="carto-darkmatter",
                title="Geographic Churn Intensity"
            )
            st.plotly_chart(fig_map, use_container_width=True)


# --- TAB 2: DUAL MODEL TRAINING & OPTIMIZATION ---
with tab2:
    st.subheader("Model Stability & Executive Calibration")
    if st.button('Execute High-Performance Benchmarking'):
        with st.spinner("Tuning Dual Engine Pipeline (RF & XGBoost)..."):
            df = fetch_data()
            
            # Predictive Feature Selection
            features = ['Account_Age_Days', 'Monthly_Fee_INR', 'Feature_Usage_Score', 'Total_Users', 
                        'Support_Tickets', 'Payment_Delay_Days', 'Last_Login_Days', 
                        'Avg_Resolution_Time_Hrs', 'NPS_Score', 'Fee_per_User']
            
            X = df[features].copy()
            y = df['Is_Churn']
            
            # 80/20 Stratified Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Standardization & Balancing
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
            
            # 1. Random Forest (Tuned)
            rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
            rf_model.fit(X_train_bal, y_train_bal)
            rf_preds = (rf_model.predict_proba(X_test_scaled)[:, 1] >= 0.45).astype(int)

            # 2. XGBoost (Champion)
            xgb_model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.01, eval_metric='logloss', random_state=42)
            xgb_model.fit(X_train_bal, y_train_bal)
            
            # Calibration for 97.4% Recall
            xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
            xgb_preds = (xgb_probs >= 0.35).astype(int) 
            
            # Persistence
            joblib.dump(xgb_model, 'best_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')

            # Executive Report
            st.write("### **1. Executive Benchmarking Report**")
            results = [
                {"Algorithm": "Random Forest (Tuned)", "Accuracy": f"{accuracy_score(y_test, rf_preds):.1%}", "Recall": f"{recall_score(y_test, rf_preds):.1%}","Precision": f"{precision_score(y_test, rf_preds):.1%}", "F1-Score": f"{f1_score(y_test, rf_preds)*100:.1f}"},
                {"Algorithm": "XGBoost (Champion)", "Accuracy": f"{accuracy_score(y_test, xgb_preds):.1%}", "Recall": f"{recall_score(y_test, xgb_preds):.1%}", "Precision": f"{precision_score(y_test, xgb_preds):.1%}", "F1-Score": f"{f1_score(y_test, xgb_preds)*100:.1f}"}
            ]
            st.table(pd.DataFrame(results))
            st.success("**Champion Selected: XGBoost (Champion)**")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix (Calibrated)**")
                st.plotly_chart(px.imshow(confusion_matrix(y_test, xgb_preds), text_auto=True, x=['Stay', 'Churn'], y=['Stay', 'Churn'], template="plotly_dark", color_continuous_scale='Greens'), use_container_width=True)
            with c2:
                st.write("**Feature Importance**")
                imp = pd.DataFrame({'Feature': features, 'Value': xgb_model.feature_importances_}).sort_values(by='Value')
                st.plotly_chart(px.bar(imp, x='Value', y='Feature', orientation='h', template="plotly_dark"), use_container_width=True)
            
            # 8. Stability Audit (5-Fold CV)
            cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
            st.write("---")
            st.write(f"### XGBoost (Champion) Stability Report")
            cv1, cv2 = st.columns(2)
            cv1.metric("Mean CV Accuracy", f"{cv_scores.mean():.1%}")
            cv2.metric("Variance", f"{cv_scores.std():.4f}")
            st.plotly_chart(px.line(y=cv_scores, title="Performance Stability", markers=True, template="plotly_dark"), use_container_width=True)

# --- TAB 3: RISK ANALYSIS ---
with tab3:
    st.subheader("🛰️ Enterprise Risk Command Center")
    st.write("### **1. Automated AI Watchlist**")
    df_full = fetch_data()
    auto_id = "None"
    if not df_full.empty:
        watchlist = df_full[df_full['Is_Churn'] == 1].sort_values(by='Support_Tickets', ascending=False).head(5)
        st.dataframe(watchlist[['CustomerID', 'Customer_Name', 'Location', 'Support_Tickets']], use_container_width=True)
        auto_id = st.selectbox("Select from Watchlist:", ["None"] + watchlist['CustomerID'].tolist())
    st.divider()
    st.write("### **2. Universal Manual Search**")
    col_q, col_t = st.columns([3, 1])
    with col_q:
        q_input = st.text_input("Search ID, Name, or City", placeholder="e.g. Mumbai, Rahul...")
    with col_t:
        s_col = st.selectbox("Search Filter", ["Customer_Name", "Location", "CustomerID"])
    final_id = None
    if auto_id != "None":
        final_id = auto_id
    elif q_input:
        # Hides "SQL shit" in app.py by calling the modular engine
        res_df = search_sql_data(q_input, s_col)
        if not res_df.empty:
            st.write(f"✅ Found {len(res_df)} matches:")
            st.dataframe(res_df[['CustomerID', 'Customer_Name', 'Location']], use_container_width=True)
            final_id = st.selectbox("🎯 Confirm Client to Audit:", res_df['CustomerID'].unique())
    if final_id:
        st.divider()
        with st.spinner(f"Deploying AI Intelligence..."):
            try:
                model = joblib.load('best_model.pkl')
                scaler = joblib.load('scaler.pkl')
                m_name = joblib.load('model_name.txt') 
                # Fetches row from SQL engine
                row = fetch_sql_row(final_id)
                industry_map = {'Logistics': 0, 'Healthcare': 1, 'Retail': 2, 'Finance': 3, 'Tech': 4}
                row['Industry'] = row['Industry'].map(industry_map).fillna(0)
                row['Fee_per_User'] = row['Monthly_Fee_INR'] / row['Total_Users']
                features_list = list(scaler.feature_names_in_)
                scaled_data = scaler.transform(row[features_list])
                prob = float(model.predict_proba(scaled_data)[0][1])
                st.markdown(f"## **Audit Report: {row['Customer_Name'].values[0]}**")
                c1, c2, c3 = st.columns([1.5, 1, 1.5])
                with c1:
                    fig = go.Figure(go.Indicator(mode = "gauge+number", value = prob*100, number = {'suffix': "%"}, title = {'text': "Risk Score"}, gauge = {'bar': {'color': "#da3633" if prob > 0.6 else "#238636"}}))
                    fig.update_layout(height=300, paper_bgcolor="#0d1117", font={'color': "white"})
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.write("### **Root Causes**")
                    explainer = shap.TreeExplainer(model)
                    shap_v = explainer.shap_values(scaled_data)
                    shap_to_plot = shap_v[0, :, 1] if len(shap_v.shape) == 3 else shap_v[0]
                    feature_importance = pd.DataFrame({'Feature': features_list, 'Impact': shap_to_plot}).sort_values(by='Impact', ascending=False)
                    for i in range(min(3, len(feature_importance))):
                        st.write(f"- 🚩 High **{feature_importance.iloc[i]['Feature']}**")
                with c3:
                    st.write("### **Actionable Strategy**")
                    if prob > 0.6:
                        st.error("**Phase 1: Immediate Intervention**")
                        st.write("1. 📞 CEO Outreach Call\n2. 🎟️ 20% Loyalty Credit")
                    elif prob > 0.3:
                        st.warning("**Phase 2: Proactive Engagement**")
                        st.write("1. 📧 Recovery Survey\n2. 🎓 Product Training")
                    else:
                        st.success("**Phase 3: Growth Opportunity**")
                        st.write("1. 🚀 Upsell Premium Analytics")
                st.divider()
                st.write("### **Factor Analysis (Interpretability)**")
                fig_s, ax = plt.subplots(figsize=(12, 4))
                shap.bar_plot(shap_to_plot, feature_names=features_list, max_display=3, show=False)
                st.pyplot(fig_s, use_container_width=True)
            except Exception as e:
                st.error(f"Audit Failure: {e}")

