import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Breast Cancer Diagnosis Predictor")
st.write("Enter tumor measurements to predict malignant vs benign.")

# Collect all 30 features
mean_radius = st.number_input("Mean Radius", value=14.0)
mean_texture = st.number_input("Mean Texture", value=19.0)
mean_perimeter = st.number_input("Mean Perimeter", value=92.0)
mean_area = st.number_input("Mean Area", value=654.0)
mean_smoothness = st.number_input("Mean Smoothness", value=0.1)
mean_compactness = st.number_input("Mean Compactness", value=0.2)
mean_concavity = st.number_input("Mean Concavity", value=0.3)
mean_concave_points = st.number_input("Mean Concave Points", value=0.1)
mean_symmetry = st.number_input("Mean Symmetry", value=0.2)
mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.06)

radius_error = st.number_input("Radius Error", value=0.4)
texture_error = st.number_input("Texture Error", value=1.0)
perimeter_error = st.number_input("Perimeter Error", value=3.0)
area_error = st.number_input("Area Error", value=40.0)
smoothness_error = st.number_input("Smoothness Error", value=0.01)
compactness_error = st.number_input("Compactness Error", value=0.02)
concavity_error = st.number_input("Concavity Error", value=0.03)
concave_points_error = st.number_input("Concave Points Error", value=0.01)
symmetry_error = st.number_input("Symmetry Error", value=0.02)
fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.003)

worst_radius = st.number_input("Worst Radius", value=16.0)
worst_texture = st.number_input("Worst Texture", value=25.0)
worst_perimeter = st.number_input("Worst Perimeter", value=110.0)
worst_area = st.number_input("Worst Area", value=880.0)
worst_smoothness = st.number_input("Worst Smoothness", value=0.15)
worst_compactness = st.number_input("Worst Compactness", value=0.25)
worst_concavity = st.number_input("Worst Concavity", value=0.35)
worst_concave_points = st.number_input("Worst Concave Points", value=0.15)
worst_symmetry = st.number_input("Worst Symmetry", value=0.3)
worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.08)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area,
                            mean_smoothness, mean_compactness, mean_concavity, mean_concave_points,
                            mean_symmetry, mean_fractal_dimension, radius_error, texture_error,
                            perimeter_error, area_error, smoothness_error, compactness_error,
                            concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                            worst_radius, worst_texture, worst_perimeter, worst_area,
                            worst_smoothness, worst_compactness, worst_concavity, worst_concave_points,
                            worst_symmetry, worst_fractal_dimension]])
    
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
    diagnosis = "Malignant" if result[0] == 0 else "Benign"
    st.success(f"Prediction: {diagnosis}")
