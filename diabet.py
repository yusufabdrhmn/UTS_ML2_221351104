import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="diabetes-classification.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("diabetes-classification")
st.write("Menentukan klasifikasi diabetes")

# Form input pengguna
Age = st.number_input("Age", min_value=50, max_value=100, value=65)
BMI = st.number_input("BMI ", min_value=24, max_value=50, value=35)
Chol= st.number_input("Chol", min_value=4.2, max_value=20.0, value=10.0)
TG = st.number_input("TG", min_value=0.9, max_value=20.0, value=10.0)
HDL= st.number_input("HDL", min_value=2.4, max_value=20.0, value=10.0)
LDL= st.number_input("LDL", min_value=1.4, max_value=20.0, value=10.0)
Cr= st.number_input("Cr", min_value=46.0, max_value=100.0, value=75.0)
BUN= st.number_input("BUN", min_value=4.7, max_value=20.0, value=10.0)

if st.button("Hasil Prediksi Penyakit Diabetes"):
    # Preprocessing input
    input_data = np.array([[Age,BMI,Chol,TG,HDL,LDL,Cr,BUN]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    label_map = {0: "Negatif", 1: "Positif"}
    crop_name = label_map[predicted_label]

    st.success(f"Prediksi Penyakit Diabetes: **{crop_name.upper()}**")
