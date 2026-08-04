# Credit Card Fraud Detection, a Risk Analysis Project

## Overview
Machine learning project built to detect fraudulent credit card transactions. Trained and evaluated on 284,807 real transactions from a European bank.

## Business Problem
Only 0.17% of transactions are fraudulent. A naive model predicting "not fraud" achieves 99.8% accuracy but catches zero fraud cases. This project addresses that imbalance and evaluates models using precision and recall which are the metrics that actually matter in a risk context.

## Models Built
- Logistic Regression (baseline)
- Random Forest (best performer with 91% precision, 83% recall)
- Gradient Boosting
- Isolation Forest (unsupervised anomaly detection)

## Key Results
- AUC Score: 0.99
- Random Forest identified 81 of 98 fraud cases in the test set
- Only 8 false positives out of 56,864 legitimate transactions
- SHAP analysis used to explain individual predictions for risk committee use

## Tech Stack
Python, Jupyter, scikit-learn, SHAP, Streamlit, pandas, seaborn

## How to Run
1. Clone the repository
2. Install requirements: pip install -r requirements.txt
3. Run the notebook: jupyter notebook
4. Launch dashboard: streamlit run app.py
