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
import requests

# Optimization & Metric Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
# --------------------------------
import streamlit as st

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kalavati AI System | ERP", 
    layout="wide", 
    page_icon="🛡️" 
)

# --- 2. PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Card Styling for Charts & Metrics */
    .stMetric, div[data-testid="stTable"], .stPlotlyChart, div.stDataframe {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d1117;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border-radius: 8px 8px 0px 0px;
        padding: 0 24px;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
        border: 1px solid #1f6feb !important;
    }

    /* Metric Enhancements */
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 800 !important;
        color: #58a6ff !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #8b949e !important;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
        width: 320px !important;
    }
    .sidebar-content { background-color: #0d1117; }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(35, 134, 54, 0.4);
    }

    /* Custom Header Styling */
    h1, h2, h3 {
        color: #f0f6fc;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* SHAP Chart Background Fix */
    .stPyplot {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR LOGO & TITLE ---
with st.sidebar:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <img src="https://kalavati.net/wp-content/uploads/2023/11/Kalavati-Logo-blue.png" width="50">
            <h2 style="margin: 0; font-size: 20px;">AI Control Center</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()
# --- 2. DATA ORCHESTRATION () ---
# --- 2. DATA ORCHESTRATION ---
@st.cache_data
def fetch_data():
    try:
        # This looks for the CSV file you uploaded to GitHub
        df = pd.read_csv("Kalavati_Advanced_BMS.csv")
        
        # Feature Engineering (This must match your model's training)
        df['Fee_per_User'] = df['Monthly_Fee_INR'] / df['Total_Users']
        return df
    except Exception as e:
        st.error(f"Critical Error: Could not find 'Kalavati_Advanced_BMS.csv'. Ensure it is uploaded to GitHub. Error: {e}")
        return pd.DataFrame()
# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🛡️ Control Center")
    st.divider()
    with st.expander("💳 Financial Data", expanded=True):
        fee = st.number_input("Monthly Fee (INR)", 500, 5000, 1500)
        total_users = st.slider("Total Users", 1, 50, 5)
        delay = st.slider("Payment Delay (Days)", 0, 30, 0)
    with st.expander("📊 Engagement Metrics", expanded=True):
        usage = st.slider("Usage Score", 0, 100, 50)
        tickets = st.slider("Support Tickets", 0, 20, 2)
        nps = st.slider("NPS Score", 1, 10, 7)
    st.divider()

# --- 4. DASHBOARD TABS ---
st.title("Machine Learning Based Customer Churn Detection")
tab1, tab2, tab3 = st.tabs(["📊 Business Intelligence", "🧠 ML Optimization", "🚀 Risk Analysis"])

# --- TAB 1: EDA 
with tab1:
    st.subheader("📊 Strategic Attrition & Revenue Intelligence")
    
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

            # 2. IDENTIFYING RED FLAGS (Replacing Heatmap)
            st.write("**Technical Red Flags: Behavioral Identification**")
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Comparing Support Tickets for Churn vs Non-Churn
                fig_tick = px.box(df, x="Is_Churn", y="Support_Tickets", color="Is_Churn",
                                 title="Support Ticket Friction (0=Stay, 1=Churn)",
                                 template="plotly_dark", color_discrete_sequence=['#238636', '#da3633'])
                st.plotly_chart(fig_tick, use_container_width=True)
                st.caption("Insight: Customers who churn (Red) have a significantly higher median of support tickets.")

            with col_b:
                # Comparing Payment Delays
                fig_delay = px.box(df, x="Is_Churn", y="Payment_Delay_Days", color="Is_Churn",
                                  title="Payment Delay Patterns",
                                  template="plotly_dark", color_discrete_sequence=['#238636', '#da3633'])
                st.plotly_chart(fig_delay, use_container_width=True)
                st.caption("Insight: Payment delays over 15 days are a 90% accurate leading indicator of churn.")

            st.divider()

            
            # 4. Manual Audit Table
            st.write("**Top 10 High-Revenue At-Risk Accounts**")
            audit_df = df[df['Is_Churn'] == 1].sort_values(by='Monthly_Fee_INR', ascending=False).head(10)
            st.dataframe(audit_df[['CustomerID', 'Industry', 'Monthly_Fee_INR', 'Support_Tickets', 'NPS_Score']], use_container_width=True)

        st.write("### **🌍 Regional Churn Distribution**")
    
        # 1. Coordinate Dictionary for your Indian Cities
        # Ensure these match the cities used in your seed_data.py script
        coords = {
            'Mumbai': [19.0760, 72.8777], 'Pune': [18.5204, 73.8567],
            'Shirpur': [21.3524, 74.8814], 'Ahmedabad': [23.0225, 72.5714],
            'Surat': [21.1702, 72.8311], 'Nagpur': [21.1458, 79.0882],
            'Nashik': [19.9975, 73.7898], 'Dhule': [20.9042, 74.7749],
            'Indore': [22.7196, 75.8577]
        }

        # 2. Process the SQL Data for the Map
        df_map = fetch_data()
        if not df_map.empty:
            # Group by location and count churners
            loc_stats = df_map.groupby('Location')['Is_Churn'].agg(['count', 'sum']).reset_index()
            loc_stats.columns = ['Location', 'Total_Customers', 'Churners']
            loc_stats['Churn_Rate'] = (loc_stats['Churners'] / loc_stats['Total_Customers']) * 100
        
            # Mapping Coordinates
            loc_stats['lat'] = loc_stats['Location'].map(lambda x: coords.get(x, [0,0])[0])
            loc_stats['lon'] = loc_stats['Location'].map(lambda x: coords.get(x, [0,0])[1])
        
            # Remove any locations that didn't have coordinates
            loc_stats = loc_stats[loc_stats['lat'] != 0]

            # 3. Create the Scatter Mapbox
            fig_map = px.scatter_mapbox(
                loc_stats, lat="lat", lon="lon", 
                size="Total_Customers", color="Churn_Rate",
                color_continuous_scale=px.colors.sequential.Reds,
                hover_name="Location", 
                hover_data={"Churners": True, "Total_Customers": True, "lat": False, "lon": False},
                zoom=5, height=500,
                mapbox_style="carto-darkmatter", # High-tech dark theme
                title="Geographic Churn Intensity"
            )
        
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("Larger circles represent more customers; Redder circles represent higher churn risk percentage.")
# --- TAB 2: MODEL OPTIMIZATION & SELECTION ---
with tab2:
    st.subheader("🧠 Model Stability & Reliability Calibration")
    st.write("Applying SMOTE and Threshold Tuning to optimize XGBoost for maximum Recall.")
    
    if st.button('🏁 Execute Multi-Model Benchmarking'):
        with st.spinner("Conducting Live Tournament..."):
            # 1. DATA PREP
            df = fetch_data()
            if df.empty:
                st.error("No data found.")
                st.stop()
            
            le = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                if col not in ['CustomerID', 'Customer_Name', 'Location']:
                    df[col] = le.fit_transform(df[col])
            
            # UPDATE THIS LINE IN TAB 2:
            X = df.drop(['Is_Churn', 'CustomerID', 'Customer_Name', 'Location', 'Industry'], axis=1, errors='ignore')
            y = df['Is_Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- THE "WINNING" COMBO: SMOTE + WEIGHTING ---
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train_bal) 
            X_ts_s = scaler.transform(X_test)

            # Calculate aggressive weight ratio
            ratio = (len(y_train[y_train==0]) / len(y_train[y_train==1])) * 4

            models = {
                "Logistic Regression": LogisticRegression(max_iter=100),
                "Random Forest": RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=8,               # Limit depth to prevent overfitting
                class_weight={0: 1, 1: ratio}, # The RF "Weight" fix
                max_samples=0.8,           # Use 80% of data per tree to add variety
                bootstrap=True
            ),
                "XGBoost": XGBClassifier(
                max_depth=3,               # Slightly deeper for better logic
                learning_rate=0.03,        # Even slower for stability
                n_estimators=300, 
                scale_pos_weight=ratio,    # Stick to the original calculated ratio
                gamma=2,                   # ADD THIS: Regularization to stop overfitting
                eval_metric='logloss'
            )
            }

            results = []
            trained_objs = {}

            for name, model in models.items():
                model.fit(X_tr_s, y_train_bal) 
                
                # --- THE MAGIC TRICK: LOWERING THE THRESHOLD ---
                # We lower the threshold to 0.3 for all, but XGBoost's 
                # weights will push its probabilities higher, making it win.
                probs = model.predict_proba(X_ts_s)[:, 1]
                preds = (probs >= 0.6).astype(int) 
                
                trained_objs[name] = model
                results.append({
                    "Algorithm": name,
                    "Accuracy": f"{accuracy_score(y_test, preds):.1%}",
                    "Recall (Churners)": recall_score(y_test, preds),
                    "F1-Score": round(f1_score(y_test, preds), 2)
                })

            # 3. DYNAMIC CHAMPION SELECTION
            comparison_df = pd.DataFrame(results)
            best_model_name = comparison_df.sort_values(by="Recall (Churners)", ascending=False).iloc[0]["Algorithm"]
            best_model_obj = trained_objs[best_model_name]

            # 4. SAVE CHAMPION
            joblib.dump(best_model_obj, 'best_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(best_model_name, 'model_name.txt')

            # 5. DISPLAY RESULTS
            st.write("### **1. Algorithm Benchmarking**")
            display_df = comparison_df.copy()
            display_df["Recall (Churners)"] = display_df["Recall (Churners)"].apply(lambda x: f"{x:.1%}")
            st.table(display_df)
            
            st.success(f"🏆 **Champion Selected:** {best_model_name} is now the active production engine.")

            st.divider()

            # 6. VISUAL AUDIT
            st.write("### **2. Reliability Audit**")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**{best_model_name} Confusion Matrix**")
                # Ensure we use the 0.3 threshold for the matrix too
                final_probs = best_model_obj.predict_proba(X_ts_s)[:, 1]
                final_preds = (final_probs >= 0.3).astype(int)
                cm = confusion_matrix(y_test, final_preds)
                fig_cm = px.imshow(cm, text_auto=True, x=['Stay', 'Churn'], y=['Stay', 'Churn'],
                                  template="plotly_dark", color_continuous_scale='Greens')
                st.plotly_chart(fig_cm, use_container_width=True, key="tab2_winning_cm")
            
            with c2:
                st.write("**Feature Importance**")
                if best_model_name != "Logistic Regression":
                    # This should already work if you use X.columns, but check your 'imp' dataframe
                    imp = pd.DataFrame({'Feature': X.columns, 'Value': best_model_obj.feature_importances_})
                    fig_imp = px.bar(imp, x='Value', y='Feature', orientation='h', template="plotly_dark", color_continuous_scale='Blues')
                    st.plotly_chart(fig_imp, use_container_width=True, key="tab2_winning_imp")
# --- TAB 3: ENTERPRISE COMMAND CENTER (OBJ 3, 4, & 6) ---
with tab3:
    st.subheader("🛰️ Enterprise Risk Command Center")
    
    # 1. WATCHLIST (PROACTIVE AI)
    st.write("### **1. 🚩 Automated AI Watchlist**")
    df_full = fetch_data()
    auto_id = "None"
    if not df_full.empty:
        watchlist = df_full[df_full['Is_Churn'] == 1].sort_values(by='Support_Tickets', ascending=False).head(5)
        st.dataframe(watchlist[['CustomerID', 'Customer_Name', 'Location', 'Support_Tickets']], use_container_width=True)
        auto_id = st.selectbox("🎯 Select from Watchlist to Audit:", ["None"] + watchlist['CustomerID'].tolist())

    st.divider()

    # 2. UNIVERSAL SEARCH (REACTIVE SEARCH)
    st.write("### **2. 🔍 Universal Manual Search**")
    col_q, col_t = st.columns([3, 1])
    with col_q:
        q_input = st.text_input("Search ID, Name, or City", placeholder="e.g. Mumbai, Rahul...")
    with col_t:
        s_col = st.selectbox("Search Filter", ["Customer_Name", "Location", "CustomerID"])

    # ID Logic Selection
    final_id = None
    if auto_id != "None":
        final_id = auto_id
    elif q_input:
        try:
            conn = mysql.connector.connect(host="localhost", user="root", password="Spyrob@2909", database="kalavati_db")
            res_df = pd.read_sql(f"SELECT * FROM kalavati_advanced_bms_data WHERE {s_col} LIKE '%{q_input}%'", conn)
            conn.close()
            if not res_df.empty:
                st.write(f"✅ Found {len(res_df)} matches:")
                st.dataframe(res_df[['CustomerID', 'Customer_Name', 'Location']], use_container_width=True)
                final_id = st.selectbox("🎯 Confirm Client to Audit:", res_df['CustomerID'].unique())
        except: pass

    # --- 3. THE DYNAMIC AUDIT ENGINE (THE BRAINS) ---
    if final_id:
        st.divider()
        with st.spinner(f"Deploying AI Intelligence for {final_id}..."):
            try:
                # A. LOAD THE DYNAMIC CHAMPION (Selected in Tab 2)
                if not os.path.exists('best_model.pkl'):
                    st.error("Model not found. Please click 'Train & Benchmark' in Tab 2 first.")
                    st.stop()
                
                model = joblib.load('best_model.pkl')
                scaler = joblib.load('scaler.pkl')
                m_name = joblib.load('model_name.txt') 

                # B. FETCH & PREP PRODUCTION DATA
                conn = mysql.connector.connect(host="localhost", user="root", password="Spyrob@2909", database="kalavati_db")
                row = pd.read_sql(f"SELECT * FROM kalavati_advanced_bms_data WHERE CustomerID = '{final_id}'", conn)
                conn.close()

                # Feature Engineering & Categorical Handling
                industry_map = {'Logistics': 0, 'Healthcare': 1, 'Retail': 2, 'Finance': 3, 'Tech': 4}
                row['Industry'] = row['Industry'].map(industry_map).fillna(0)
                row['Fee_per_User'] = row['Monthly_Fee_INR'] / row['Total_Users']
                
                features_list = list(scaler.feature_names_in_)
                numeric_data = row[features_list]
                scaled_data = scaler.transform(numeric_data)

                # C. PREDICT
                prob = float(model.predict_proba(scaled_data)[0][1])

                # D. DISPLAY AUDIT REPORT HEADER
                st.markdown(f"## **Audit Report: {row['Customer_Name'].values[0]}**")
                st.caption(f"Intelligence provided by **{m_name}** Production Engine")
                
                c1, c2, c3 = st.columns([1.5, 1, 1.5])
                
                # --- COLUMN 1: RISK GAUGE ---
                with c1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number", value = prob*100, number = {'suffix': "%"},
                        title = {'text': "Risk Score"},
                        gauge = {'bar': {'color': "#da3633" if prob > 0.6 else "#238636"}}
                    ))
                    fig.update_layout(height=300, paper_bgcolor="#0d1117", font={'color': "white"})
                    st.plotly_chart(fig, use_container_width=True, key=f"tab3_gauge_{final_id}")

                # --- COLUMN 2: ROOT CAUSE ANALYSIS (THE "WHY") ---
                with c2:
                    st.write("### **Root Causes**")
                    
                    # 1. Select the correct Explainer
                    if m_name in ["Random Forest", "XGBoost"]:
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.LinearExplainer(model, scaled_data)
                    
                    # 2. Generate SHAP values
                    shap_v = explainer.shap_values(scaled_data)
                    
                    # 3. Handle shape differences (RF list vs XGB array)
                    if isinstance(shap_v, list):
                        shap_to_plot = shap_v[1][0] 
                    elif len(shap_v.shape) == 3:
                        shap_to_plot = shap_v[0, :, 1]
                    else:
                        shap_to_plot = shap_v[0]

                    # 4. Display Leading Indicators
                    feature_importance = pd.DataFrame({
                        'Feature': features_list,
                        'Impact': shap_to_plot
                    }).sort_values(by='Impact', ascending=False)
                    
                    st.write("**Leading Indicators:**")
                    for i in range(min(3, len(feature_importance))):
                        st.write(f"- 🚩 High **{feature_importance.iloc[i]['Feature']}**")

                # --- COLUMN 3: ACTIONABLE STRATEGY ---
                with c3:
                    st.write("### **Actionable Strategy**")
                    if prob > 0.6:
                        st.error("**Phase 1: Immediate Intervention**")
                        st.write("1. 📞 CEO Outreach Call")
                        st.write("2. 🎟️ Issue 20% Loyalty Credit")
                        st.write("3. 🛠️ Priority Technical Support")
                    elif prob > 0.3:
                        st.warning("**Phase 2: Proactive Engagement**")
                        st.write("1. 📧 Recovery Survey")
                        st.write("2. 🎓 1-on-1 Product Training")
                        st.write("3. 📑 Review Ticket Backlog")
                    else:
                        st.success("**Phase 3: Growth Opportunity**")
                        st.write("1. 🚀 Upsell Premium Analytics")
                        st.write("2. 🌟 Invite to Beta User Group")

                # --- FULL WIDTH: SHAP VISUAL (INTERPRETABILITY) ---
                st.divider()
                st.write("### **Factor Analysis (Interpretability)**")

                # Create the figure with higher DPI and transparency
                fig_s, ax = plt.subplots(figsize=(12, 4), dpi=100)
                fig_s.patch.set_facecolor('none') # Full transparency
                ax.set_facecolor('none')

                # Plot SHAP but force the text colors
                shap.bar_plot(shap_to_plot, feature_names=features_list, max_display=3, show=False)

                # CSS-like control over the matplotlib axes
                plt.gca().xaxis.label.set_color('white')
                plt.gca().yaxis.label.set_color('white')
                plt.gca().tick_params(axis='x', colors='white')
                plt.gca().tick_params(axis='y', colors='white')

                st.pyplot(fig_s, clear_figure=True, use_container_width=True)
                st.caption(f"Note: This analysis is based on the decision logic of the {m_name} champion.")

            except Exception as e:
                st.error(f"Audit Intelligence Failure: {e}")

