import streamlit as st
import pandas as pd
import numpy as np

st.title('Credit Card Fraud Detection')
st.subheader('Risk Analysis Dashboard')
st.write('Random Forest model trained on 284,807 transactions')

# Model metrics
st.header('Model Performance')
col1, col2, col3 = st.columns(3)
col1.metric('Precision', '91%')
col2.metric('Recall', '83%')
col3.metric('AUC Score', '0.99')

# Dataset Overview
st.header('Dataset Overview')
col1, col2, col3 = st.columns(3)
col1.metric('Total Transactions', '284,807')
col2.metric('Fraudulent', '492 (0.17%)')
col3.metric('Legitimate', '284,315')

# Model Comparison
st.header('Model Comparison')
model_data = {
    'Model': ['Random Forest', 'Logistic Regression', 'Gradient Boosting', 'Isolation Forest'],
    'Precision': ['91%', '6%', '12%', '32%'],
    'Recall': ['83%', '93%', '91%', '36%'],
    'AUC Score': ['0.99', '0.97', '0.98', 'N/A']
}
st.dataframe(pd.DataFrame(model_data), hide_index=True, use_container_width=True)

# Confusion Matrix Results
st.header('Random Forest Confusion Matrix Results:')
col1, col2 = st.columns(2)
with col1:
    st.success('Correctly identified legitimate: 56,856')
    st.success('Fraud cases caught: 81')
with col2:
    st.error('Missed fraud cases: 17')
    st.warning('False alarms: 8')

# Business Cost Analysis
st.header('Business Cost Analysis')
missed_fraud = 17
false_alarms = 8
avg_fraud_amount = 122.21

col1, col2 = st.columns(2)
with col1:
    st.metric('Estimated Financial Exposure (missed fraud)', f'£{round(missed_fraud * avg_fraud_amount, 2):,}')
with col2:
    st.metric('Analyst Review Cost (false alarms)', f'£{false_alarms * 15}')

st.info('Assumption: average fraud transaction £122.21 and analyst review cost £15 per case (30 mins at £30/hr)')

# Key Findings
st.header('Key Findings')
st.write("""
- **Random Forest** outperformed all other models, achieving 91% precision and 83% recall
- Class imbalance (0.17% fraud) was addressed using SMOTE oversampling
- SHAP analysis identified **V14, V12, and V4** as the most important features for fraud detection
- The model catches 83% of fraud while keeping false alarms extremely low (8 cases in 56,962 test transactions)
- Financial exposure from missed fraud: **£2,077** vs potential loss without any model: **£60,247**
""")
