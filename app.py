
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Breast Cancer Diagnosis Predictor.")
st.write("Enter tumor measurements to predict malignant vs benign.")

# Collect 12 key features (same order as training)
mean_radius = st.number_input("Mean Radius", value=14.0)
mean_texture = st.number_input("Mean Texture", value=19.0)
mean_perimeter = st.number_input("Mean Perimeter", value=92.0)
mean_area = st.number_input("Mean Area", value=654.0)
mean_smoothness = st.number_input("Mean Smoothness", value=0.1)
mean_compactness = st.number_input("Mean Compactness", value=0.2)
mean_concavity = st.number_input("Mean Concavity", value=0.3)
mean_concave_points = st.number_input("Mean Concave Points", value=0.1)

worst_radius = st.number_input("Worst Radius", value=16.0)
worst_texture = st.number_input("Worst Texture", value=25.0)
worst_perimeter = st.number_input("Worst Perimeter", value=110.0)
worst_area = st.number_input("Worst Area", value=880.0)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area,
                            mean_smoothness, mean_compactness, mean_concavity, mean_concave_points,
                            worst_radius, worst_texture, worst_perimeter, worst_area]])
    
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)

# Debug line to check raw output (optional)
    st.write("Raw prediction value:", result[0])
    
    # Probability output
    proba = model.predict_proba(scaled_input)[0]
    st.write(f"Probability (Malignant): {proba[0]:.2f}")
    st.write(f"Probability (Benign): {proba[1]:.2f}")
    

    # Map prediction to diagnosis
    diagnosis = "Malignant" if result[0] == 0 else "Benign"
    st.success(f"Prediction: {diagnosis}")
   

