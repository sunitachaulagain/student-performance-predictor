import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('../models/student_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

st.title("Student Performance Prediction")

age = st.slider("Age", 15, 22, 18)
studytime = st.slider("Study Time", 1, 4, 2)
failures = st.slider("Failures", 0, 4, 0)
absences = st.slider("Absences", 0, 50, 5)
G1 = st.slider("G1 Grade", 0, 20, 10)
G2 = st.slider("G2 Grade", 0, 20, 10)

input_data = pd.DataFrame({
    'age': [age],
    'studytime': [studytime],
    'failures': [failures],
    'absences': [absences],
    'G1': [G1],
    'G2': [G2]
})

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

if st.button("Predict"):
    st.success(f"Predicted Final Grade (G3): {prediction[0]:.2f}")