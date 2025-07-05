import streamlit as st
import numpy as np
import joblib

model = joblib.load('wine_quality_model.pkl')

st.title("Wine Quality Predictor")
st.write("Input the chemical properties of a wine sample to predict if it is **Good** or **Not Good**.")

fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=0.00)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.00)
citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.00)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=0.00)
chlorides = st.number_input("Chlorides", min_value=0.0, value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, value=0.00)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, value=0.00)
density = st.number_input("Density", min_value=0.0, value=0.00)
pH = st.number_input("pH", min_value=0.0, value=0.00)
sulphates = st.number_input("Sulphates", min_value=0.0, value=0.00)
alcohol = st.number_input("Alcohol", min_value=0.0, value=0.00)

if st.button("Predict"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction]

    result = "🍷 Good Quality Wine" if prediction == 1 else "⚠️ Not Good Quality Wine"
    if prediction == 1:
        st.success(f"**{result}** with a confidence of **{confidence:.1%}**")
    else:
        st.error(f"**{result}** with a confidence of **{confidence:.1%}**")
