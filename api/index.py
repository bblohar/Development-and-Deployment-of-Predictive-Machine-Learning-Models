# api/index.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

class CustomerData(BaseModel):
    Account_Age_Days: int
    Monthly_Fee_INR: float
    Total_Users: int
    Feature_Usage_Score: int
    Support_Tickets: int
    Payment_Delay_Days: int
    Last_Login_Days: int
    Avg_Resolution_Time_Hrs: int
    NPS_Score: int
    Industry: int
    Subscription_Type: int

@app.get("/")
def home():
    return {"status": "Kalavati AI API is Live"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to DataFrame
    input_dict = data.dict()
    input_dict['Fee_per_User'] = input_dict['Monthly_Fee_INR'] / input_dict['Total_Users']
    
    input_df = pd.DataFrame([input_dict])[scaler.feature_names_in_]
    scaled_input = scaler.transform(input_df)
    
    # Generate prediction
    prob = float(model.predict_proba(scaled_input)[0][1])
    return {
        "churn_probability": round(prob, 4),
        "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
    }