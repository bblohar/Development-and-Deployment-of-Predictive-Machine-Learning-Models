import streamlit as st
import pandas as pd
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap

# Optimization & Metric Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

# --- 1. PAGE CONFIG & PREMIUM STYLING ---
st.set_page_config(page_title="Kalavati AI | Enterprise Churn", layout="wide", page_icon="🛡️")

# Professional CSS Injection
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #58a6ff !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #161b22; border-radius: 5px 5px 0 0; gap: 1px; padding-left: 20px; padding-right: 20px; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb !important; color: white !important; }
    div.stButton > button:first-child { background-color: #238636; color: white; border: none; width: 100%; border-radius: 5px; height: 3em; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ORCHESTRATION ---
@st.cache_data
def fetch_data():
    conn = mysql.connector.connect(
        host="localhost", user="root", password="Spyrob@2909", database="kalavati_db"
    )
    df = pd.read_sql("SELECT * FROM kalavati_db.kalavati_advanced_bms_data", conn)
    conn.close()
    df['Fee_per_User'] = df['Monthly_Fee_INR'] / df['Total_Users']
    return df

# --- 3. SIDEBAR (CONTROL CENTER) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Admin Simulator")
    st.write("Adjust customer parameters to test the model in real-time.")
    st.divider()
    
    # Grouping inputs for a cleaner look
    with st.expander("💳 Financial Data", expanded=True):
        fee = st.number_input("Monthly Fee (INR)", 500, 5000, 1500)
        total_users = st.slider("Total Users", 1, 50, 5)
        delay = st.slider("Payment Delay (Days)", 0, 30, 0)
    
    with st.expander("📊 Engagement Metrics", expanded=True):
        age = st.slider("Account Age (Days)", 30, 1000, 365)
        usage = st.slider("Usage Score", 0, 100, 50)
        tickets = st.slider("Support Tickets", 0, 20, 2)
        nps = st.slider("NPS Score", 1, 10, 7)
    
    st.divider()
    st.caption("Developed by Bhavesh Lohar | MPSTME")

# --- 4. DASHBOARD HEADER ---
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.title("🛡️ Enterprise Churn Prediction System")
    st.markdown("##### **Project Stage:** Month 6 - Final Deployment & Explainable AI")
with col_h2:
    st.image("https://kalavatitechnologies.com/wp-content/uploads/2021/04/Kalavati-Technologies-Logo.png", width=150)

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Business Intelligence", "🧠 ML Optimization", "🚀 Risk Analysis"])

# --- TAB 1: EDA ---
with tab1:
    st.subheader("Historical Performance Insights")
    if st.button('🔄 Refresh Global Data'):
        with st.spinner("Analyzing Database..."):
            df = fetch_data()
            
            # Key Stats Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers", len(df))
            m2.metric("Avg Churn Rate", f"{(df['Is_Churn'].mean()*100):.1f}%")
            m3.metric("Avg NPS", round(df['NPS_Score'].mean(), 1))
            m4.metric("Revenue at Risk", f"₹{df[df['Is_Churn']==1]['Monthly_Fee_INR'].sum():,}")

            st.write("---")
            col_eda1, col_eda2 = st.columns([2, 1])
            with col_eda1:
                fig = px.box(df, x="Is_Churn", y="Fee_per_User", color="Is_Churn", 
                             points="all", title="Fee per User Distribution by Churn Status",
                             template="plotly_dark", color_discrete_sequence=['#238636', '#da3633'])
                st.plotly_chart(fig, use_container_width=True)
            with col_eda2:
                st.write("**Recent High-Risk Activity**")
                st.dataframe(df[df['Is_Churn']==1].tail(10)[['CustomerID', 'Monthly_Fee_INR', 'Support_Tickets']], height=300)

# --- TAB 2: OPTIMIZATION ---
with tab2:
    st.subheader("Model Stability & Calibration")
    if st.button('🏁 Re-calibrate Optimized XGBoost'):
        with st.spinner("Running Hyperparameter Tuning..."):
            df = fetch_data()
            le = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                if col != 'CustomerID': df[col] = le.fit_transform(df[col])
            
            X = df.drop(['Is_Churn', 'CustomerID'], axis=1)
            y = df['Is_Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss'), 
                                       {'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}, cv=3)
            grid_search.fit(X_train_scaled, y_train)
            best_xgb = grid_search.best_estimator_
            cv_scores = cross_val_score(best_xgb, X_train_scaled, y_train, cv=5)
            
            joblib.dump(best_xgb, 'churn_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')

            # Performance Visualization
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("**Core Performance Metrics**")
                st.metric("Model Stability (CV)", f"{cv_scores.mean()*100:.2f}%")
                st.metric("F1-Score (Catch Rate)", f"{f1_score(y_test, best_xgb.predict(X_test_scaled)):.2f}")
            
            with c2:
                y_probs = best_xgb.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC: {auc(fpr, tpr):.2f})',
                                  labels={'x':'False Positives', 'y':'True Positives'}, template="plotly_dark")
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc, use_container_width=True)

# --- TAB 3: ANALYSIS ---
with tab3:
    st.subheader("Individual Risk Assessment")
    if os.path.exists('churn_model.pkl'):
        if st.button("🔮 Generate Intelligence Report"):
            model, scaler = joblib.load('churn_model.pkl'), joblib.load('scaler.pkl')
            expected_features = list(scaler.feature_names_in_)
            
            input_dict = {
                'Account_Age_Days': age, 'Monthly_Fee_INR': fee, 'Total_Users': total_users,
                'Feature_Usage_Score': usage, 'Support_Tickets': tickets, 'Payment_Delay_Days': delay,
                'Last_Login_Days': 10, 'Avg_Resolution_Time_Hrs': 5, 'NPS_Score': nps,
                'Industry': 1, 'Subscription_Type': 1, 'Fee_per_User': fee / total_users
            }
            
            input_df = pd.DataFrame([input_dict])[expected_features]
            scaled_input = scaler.transform(input_df)
            prob = float(model.predict_proba(scaled_input)[0][1])

            # Result Dashboard
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.container()
                st.write("**Risk Probability**")
                
                # Interactive Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk %", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "#da3633" if prob > 0.5 else "#238636"},
                        'steps': [
                            {'range': [0, 30], 'color': "#161b22"},
                            {'range': [30, 70], 'color': "#161b22"},
                            {'range': [70, 100], 'color': "#161b22"}]}))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="#0d1117", font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.write("**Explainable AI (SHAP): Why this prediction?**")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(scaled_input)
                
                # High-quality Matplotlib SHAP plot
                fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                fig_shap.patch.set_facecolor('#0d1117')
                ax_shap.set_facecolor('#0d1117')
                shap.bar_plot(shap_values[0], feature_names=expected_features, max_display=5, show=False)
                plt.xlabel("Impact on Churn (Red = Increases Risk)", color='white')
                plt.yticks(color='white')
                plt.xticks(color='white')
                st.pyplot(fig_shap)
                
            st.divider()
            # Actionable Intelligence
            if prob > 0.7:
                st.error("🚨 **CRITICAL RISK:** Immediate intervention required. Suggested action: Offer 20% discount on renewal.")
            elif prob > 0.4:
                st.warning("⚠️ **ELEVATED RISK:** Customer shows signs of dissatisfaction. Suggested action: Schedule a feedback call.")
            else:
                st.success("✅ **STABLE ACCOUNT:** High engagement detected. Suggested action: Upsell new premium features.")
    else:
        st.warning("Please run Model Optimization (Tab 2) to initialize the system.")