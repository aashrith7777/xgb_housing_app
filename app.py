import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and feature columns
model = joblib.load("model/xgb_house_model.joblib")
feature_columns = joblib.load("model/feature_columns.joblib")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write("Estimate the price of a house based on simple inputs.")

st.subheader("Enter House Details:")

# User-friendly inputs using sliders and simple names
median_income = st.slider("Average Family Income (₹ Lakh/year)", 1, 20, 5)
house_age = st.slider("House Age (Years)", 1, 50, 20)
average_rooms = st.slider("Average Rooms per House", 2, 10, 6)
average_bedrooms = st.slider("Average Bedrooms per House", 1, 5, 2)
population = st.slider("Population in the Area", 100, 5000, 1000)
average_occupancy = st.slider("Average People per House", 1, 10, 3)
latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

# Automatically calculate derived features
rooms_per_household = average_rooms / (house_age + 1)
bedrooms_per_room = average_bedrooms / (average_rooms + 1e-6)

# Prepare input dataframe
input_data = pd.DataFrame([[
    median_income, house_age, average_rooms, average_bedrooms,
    population, average_occupancy, latitude, longitude,
    rooms_per_household, bedrooms_per_room
]], columns=feature_columns)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    # Convert to Indian Rupees (assuming 1 unit = 1 lakh USD) as example
    prediction_inr = prediction * 100000 * 82  # Approx USD to INR
    st.success(f"Estimated House Price: ₹{prediction_inr:,.0f}")

# Optional: show feature importance
if st.checkbox("Show Feature Importance"):
    import matplotlib.pyplot as plt
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    st.bar_chart(imp_df.set_index('Feature'))
