import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
from ann_model import ANN_regressor

# Load saved scaler, encoder, and model
try:
    scaler = joblib.load("standard_scalar.joblib")
    
except FileNotFoundError:
    st.error("Scaler file 'standard_scalar.joblib' not found. Please ensure the scaler is saved in the same directory.")
    st.stop()

try:
    ohe_dict = joblib.load("ohe_dict.joblib")
except FileNotFoundError:
    st.error("encoder file 'ohe_dict' not found. Please ensure the encoder is saved in the same directory.")
    st.stop()

# Load model
default_input_size = 74 #we aready know this info from our ipynb file.
model_mse  = ANN_regressor(input_size=default_input_size)
model_huber = ANN_regressor(input_size=default_input_size)

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_mse.load_state_dict(torch.load("best_regression_model_mse.pth",map_location = torch.device(device)))
    model_mse.eval()  #Setting the model to evaluation mode!
    
except FileNotFoundError:
    st.error("Model file 'best_regression_model_mse.pth' not found. Please ensure the model is saved in the same directory.")
    st.stop()

try:
    model_huber.load_state_dict(torch.load("best_regression_model_mse.pth",map_location = torch.device(device)))
    model_huber.eval()  #Setting the model to evaluation mode!
    
except FileNotFoundError:
    st.error("Model file 'best_regression_model_huber.pth' not found. Please ensure the model is saved in the same directory.")
    st.stop()


# Load model with best svaed weights
model_huber.load_state_dict(torch.load("best_regression_model_huber.pth"))
model_huber.eval()

model_mse.load_state_dict(torch.load('best_regression_model_mse.pth'))
model_mse.eval()

# Input features
st.title("📦 Demand Prediction App")

st.header("🧾 Product & Sales Details")

# ----------------------------
# 1. DATE & TIME FEATURES
# ----------------------------
selected_date = st.date_input("Date", datetime.today())
day = selected_date.day
month = selected_date.month
year = selected_date.year
is_weekend = selected_date.weekday() >= 5

# ----------------------------
# 2. CATEGORICAL FEATURES
# ----------------------------
sku = st.selectbox("SKU", ohe_dict['ohe_sku'].categories_[0])
brand = st.selectbox("Brand", ohe_dict['ohe_brand'].categories_[0])
segment = st.selectbox("Segment", ohe_dict['ohe_segment'].categories_[0])
category = st.selectbox("Category", ohe_dict['ohe_category'].categories_[0])
pack_type = st.selectbox("Pack Type", ohe_dict['ohe_pack_type'].categories_[0])
channel = st.selectbox("Channel", ['Retail','Discount','E-commerce'])  #We actually skip this metric for prediction, just including to make it fit well with user available data
region = st.selectbox("Region",[f'PL_{x}' for x in ['Central','North','South']])   #same as above

# ----------------------------
# 3. NUMERIC FEATURES
# ----------------------------
# Here inputs are initialized close to avg mean of the entire data
price_unit = st.number_input("Unit Price", min_value=0.0, value=5.25)
promotion_flag = st.selectbox("Promotion Flag", [0, 1])
delivery_days = st.number_input("Delivery Days", min_value=0, value=3)
stock_available = st.number_input("Stock Available", min_value=0, value=150)
delivered_qty = st.number_input("Delivered Quantity", min_value=0, value=180)

# ----------------------------
# 4. Assemble Inputs
# ----------------------------
input_dict = {
    
    "sku": sku,
    "brand": brand,
    "segment": segment,
    "category": category,
    "pack_type": pack_type,
    
    "price_unit": price_unit,
    "promotion_flag": promotion_flag,
    "delivery_days": delivery_days,
    "stock_available": stock_available,
    "delivered_qty": delivered_qty,
    
    "day": day,
    "month": month,
    "year": year,
    "is_weekend": is_weekend,
    
}

df = pd.DataFrame([input_dict])
# st.write(df)
# st.write(ohe_dict)
# st.write(df.columns)

# categorical to numerical , ohe using ohe_dict file 
ohe_variables = ['sku', 'brand', 'segment', 'category', 'pack_type'] 

for column in ohe_variables:
    ohe = ohe_dict[f'ohe_{column}']
    # st.write(ohe.transform([df[column]]))
    df_var = ohe.transform([df[column]])
    df_var = pd.DataFrame(df_var,columns = ohe.get_feature_names_out([f'{column}']))
    df = pd.merge(df,df_var,left_index=True,right_index=True)
    df.drop(f'{column}',axis = 1,inplace = True)
st.write("Input Data Frame : \n")
st.write(df)
# st.write(df.columns)
df = scaler.transform(df)
st.write("Data Frame after standardization :")
st.write(df)

# Convert to PyTorch tensor
input_tensor = torch.tensor(df, dtype=torch.float32)
st.write("Input Tensor :")
st.write(input_tensor)

if st.button("Predict"):
# Model predictions 
    with torch.no_grad():
        output_mse = model_mse(input_tensor)
        output_huber = model_huber(input_tensor)
        
        st.write(f"Device in use : {device}")
        
    # Display the following results
    st.header("Prediction Results :")
    st.write(f"ANN model with Huber loss | predicted Output_sales : {output_huber[0][0]} ~ {np.round(output_huber[0][0],0)} ")
    st.write(f"ANN model with MSE loss   | predicted Output_sales : {output_mse[0][0]}   ~ {np.round(output_mse[0][0],0)} ")