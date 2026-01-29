import streamlit as st
import joblib
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Load the model
# We use a relative path assuming the model is in a 'model' folder
model_path = os.path.join('model', 'house_price_model.pkl')

@st.cache_resource
def load_model():
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please check directory structure.")
        return None

model = load_model()

# App Title and Description
st.title("🏡 House Price Prediction System")
st.write("Enter the house details below to estimate the sale price.")
st.write("---")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
        total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=6000, value=1000)
    
    with col2:
        garage_cars = st.selectbox("Garage Size (Cars)", [0, 1, 2, 3, 4])
        full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3])
        year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000)
    
    # Submit Button
    submitted = st.form_submit_button("Predict Price")

# Prediction Logic
if submitted and model:
    # Prepare input data in the same order as training
    # ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
    input_data = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]])
    
    prediction = model.predict(input_data)
    
    st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")