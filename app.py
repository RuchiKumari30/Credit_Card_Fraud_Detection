import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('fraud_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Credit Card Fraud Detection")

# List of all features in order
feature_names = [f'V{i}' for i in range(1, 29)]
feature_names = ['Time'] + feature_names + ['Amount']

# Create input fields for all 30 features
inputs = []
for feature in feature_names:
    value = st.text_input(f"{feature}", value="0.0")
    inputs.append(value)

if st.button("Check for Fraud"):
    try:
        # Convert inputs to float and reshape for model
        input_data = np.array([float(i) for i in inputs]).reshape(1, -1)

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        # Output
        if prediction[0] == 1:
            st.error("Fraudulent Transaction Detected!")
        else:
            st.success("Legitimate Transaction")
    except Exception as e:
        st.error(f"Invalid input: {e}")
