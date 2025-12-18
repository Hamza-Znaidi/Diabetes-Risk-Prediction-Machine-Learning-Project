import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("output/diabetes_model.pkl")

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("🩺 Diabetes Risk Prediction")
st.write("Enter the patient data below and get an instant risk prediction.")

st.divider()

# Input fields
num_preg = st.number_input("Number of Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Concentration", 0, 250, 120)
bp = st.slider("Diastolic Blood Pressure", 0, 140, 70)
thickness = st.slider("Skin Thickness", 0, 99, 20)
insulin = st.slider("Insulin Level", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
diab_pred = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.slider("Age", 10, 90, 33)
skin = st.slider("Skin (Custom Feature)", 0.0, 3.0, 1.0)

# Convert to dataframe
input_data = pd.DataFrame([{
    "num_preg": num_preg,
    "glucose_conc": glucose,
    "diastolic_bp": bp,
    "thickness": thickness,
    "insulin": insulin,
    "bmi": bmi,
    "diab_pred": diab_pred,
    "age": age,
    "skin": skin
}])

if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠ High Risk of Diabetes\n**Probability: {proba:.2f}**")
    else:
        st.success(f"✔ Low Risk of Diabetes\n**Probability: {proba:.2f}**")
