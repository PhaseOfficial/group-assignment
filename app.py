import streamlit as st
import pickle
import numpy as np

# Load the XGBoost model
model_path = "xgboost_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# App title
st.title("Loan Outcome Prediction")

# Input fields for features
st.write("### Input the Features:")

# Text input for optional identifiers
ID = st.text_input("ID (Optional)")
customer_id = st.text_input("Customer ID (Optional)")
country_id = st.text_input("Country ID (Optional)")
tbl_loan_id = st.text_input("Loan ID (Optional)")
lender_id = st.text_input("Lender ID (Optional)")

# Loan type (categorical feature)
loan_type = st.selectbox("Loan Type", options=["Type 1", "Type 2", "Type 3"])  # Adjust options as needed
loan_type_mapping = {"Type 1": 0, "Type 2": 1, "Type 3": 2}  # Map to numeric values

# Numerical inputs for loan amounts
Total_Amount = st.number_input("Total Amount", min_value=0.0, step=0.01)
Total_Amount_to_Repay = st.number_input("Total Amount to Repay", min_value=0.0, step=0.01)

# Disbursement date components
st.write("#### Disbursement Date")
disbursement_date_year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
disbursement_date_month = st.number_input("Month", min_value=1, max_value=12, step=1)
disbursement_date_day = st.number_input("Day", min_value=1, max_value=31, step=1)

# Due date components
st.write("#### Due Date")
due_date_year = st.number_input("Year", min_value=1900, max_value=2100, step=1, key="due_date_year")
due_date_month = st.number_input("Month", min_value=1, max_value=12, step=1, key="due_date_month")
due_date_day = st.number_input("Day", min_value=1, max_value=31, step=1, key="due_date_day")

# Additional numerical inputs
duration = st.number_input("Duration (days)", min_value=1, step=1)
New_versus_Repeat = st.selectbox("New or Repeat Borrower", options=["New", "Repeat"])
repeat_mapping = {"New": 0, "Repeat": 1}  # Map to numeric values

Amount_Funded_By_Lender = st.number_input("Amount Funded By Lender", min_value=0.0, step=0.01)
Lender_portion_Funded = st.number_input("Lender Portion Funded", min_value=0.0, max_value=1.0, step=0.01)
Lender_portion_to_be_repaid = st.number_input("Lender Portion to be Repaid", min_value=0.0, step=0.01)

# Button to make a prediction
if st.button("Predict"):
    try:
        # Prepare features for prediction
        features = np.array([
            loan_type_mapping[loan_type],  # Categorical mapping for loan type
            Total_Amount,
            Total_Amount_to_Repay,
            disbursement_date_year,
            disbursement_date_month,
            disbursement_date_day,
            due_date_year,
            due_date_month,
            due_date_day,
            duration,
            repeat_mapping[New_versus_Repeat],  # Categorical mapping for repeat borrower
            Amount_Funded_By_Lender,
            Lender_portion_Funded,
            Lender_portion_to_be_repaid
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Display result
        st.write(f"### Predicted Loan Outcome: {'Approved' if prediction[0] == 1 else 'Rejected'}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")
