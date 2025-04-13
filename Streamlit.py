import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Real Estate Price Estimator")

st.header("🏠 Real Estate Value Estimator")
st.markdown("""
Welcome to the property value estimator. Fill in the details about a home, 
and this app will estimate its selling price based on a pre-trained model.
""")

# Load trained model
model = joblib.load("property_price_model.pkl")

with st.form("form_input"):
    st.subheader("📋 Property Specifications")

    year_of_sale = st.number_input("Year of Sale", min_value=1990, max_value=2025, value=2021)
    tax_amount = st.number_input("Annual Property Tax ($)", min_value=50, max_value=5000, value=600)
    insurance_cost = st.number_input("Insurance Estimate ($)", min_value=20, max_value=2000, value=250)
    total_bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    total_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    area_sqft = st.number_input("Built Area (sqft)", min_value=300, max_value=10000, value=1800)
    construction_year = st.number_input("Year Constructed", min_value=1800, max_value=2025, value=2005)
    land_area = st.number_input("Lot Area (sqft)", min_value=0, max_value=500000, value=4000)
    has_basement = st.selectbox("Basement Included?", ["No", "Yes"])
    is_bungalow = st.selectbox("Bungalow Type?", ["No", "Yes"])
    is_condo = st.selectbox("Condo Type?", ["No", "Yes"])
    in_popular_zone = st.selectbox("Located in Popular Neighborhood?", ["No", "Yes"])
    estimated_age = st.number_input("Estimated Property Age (years)", min_value=0, max_value=200, value=15)

    submit = st.form_submit_button("Estimate Value")

if submit:
    binary_inputs = {
        "Yes": 1,
        "No": 0
    }

    inputs = np.array([
        year_of_sale,
        tax_amount,
        insurance_cost,
        total_bedrooms,
        total_bathrooms,
        area_sqft,
        construction_year,
        land_area,
        binary_inputs[has_basement],
        binary_inputs[is_bungalow],
        binary_inputs[is_condo],
        binary_inputs[in_popular_zone],
        estimated_age
    ]).reshape(1, -1)

    estimated_value = model.predict(inputs)[0]

    st.subheader("💰 Predicted Market Price:")
    st.success(f"${estimated_value:,.2f}")

st.markdown("### 🔍 Model Insights")
st.image("feature_importance.png", caption="Model Feature Contributions", use_column_width=True)
