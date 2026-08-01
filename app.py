import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Credit Card Fraud Detection')
st.subheader('Risk Analysis Dashboard')
st.write('Random Forest model trained on 284,807 transactions')

# Model metrics
st.header('Model Performance')
col1, col2, col3 = st.columns(3)
col1.metric('Precision', '91%')
col2.metric('Recall', '83%')
col3.metric('AUC Score', '0.99')

# Transaction Risk Checker
st.header('Transaction Risk Checker')
st.write('Adjust the values below to simulate a transaction:')

amount = st.slider('Transaction Amount (£)', 0.0, 25000.0, 100.0)
v12 = st.slider('V12', -20.0, 20.0, 0.0)
v14 = st.slider('V14', -20.0, 20.0, 0.0)
v4 = st.slider('V4', -20.0, 20.0, 0.0)

if st.button('Check Transaction'):
    input_data = np.zeros((1, 30))
    input_data[0, 0] = 0
    input_data[0, 11] = v12
    input_data[0, 13] = v14
    input_data[0, 3] = v4
    input_data[0, 29] = amount

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f'HIGH RISK — Fraud probability: {round(probability * 100, 1)}%')
    else:
        st.success(f'LOW RISK — Fraud probability: {round(probability * 100, 1)}%')