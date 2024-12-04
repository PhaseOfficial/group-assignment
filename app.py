import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the XGBoost model
def load_model():
    with open("xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Feature input function
def user_input():
    st.sidebar.header("Input Features")
    customer_id = st.sidebar.number_input("Customer ID", min_value=0, value=1)
    tbl_loan_id = st.sidebar.number_input("Loan ID", min_value=0, value=1)
    lender_id = st.sidebar.number_input("Lender ID", min_value=0, value=1)
    Total_Amount = st.sidebar.number_input("Total Amount", value=1000.0)
    Total_Amount_to_Repay = st.sidebar.number_input("Total Amount to Repay", value=1200.0)
    duration = st.sidebar.number_input("Duration (days)", value=365)
    New_versus_Repeat = st.sidebar.selectbox("New versus Repeat (0: New, 1: Repeat)", [0, 1])
    Amount_Funded_By_Lender = st.sidebar.number_input("Amount Funded By Lender", value=500.0)
    Lender_portion_Funded = st.sidebar.number_input("Lender Portion Funded", value=50.0)
    Lender_portion_to_be_repaid = st.sidebar.number_input("Lender Portion to be Repaid", value=60.0)
    
    disbursement_date = st.sidebar.date_input("Disbursement Date", pd.to_datetime("2023-01-01"))
    due_date = st.sidebar.date_input("Due Date", pd.to_datetime("2023-12-31"))
    
    loan_type = st.sidebar.selectbox(
        "Loan Type",
        ["Type_1", "Type_2", "Type_3", "Type_4", "Type_5", "Type_6", "Type_7", "Type_8", 
         "Type_9", "Type_10", "Type_11", "Type_12", "Type_13", "Type_14", "Type_15", 
         "Type_16", "Type_17", "Type_18", "Type_19", "Type_20", "Type_21", "Type_22", 
         "Type_23", "Type_24"]
    )
    
    data = {
        "customer_id": customer_id,
        "tbl_loan_id": tbl_loan_id,
        "lender_id": lender_id,
        "Total_Amount": Total_Amount,
        "Total_Amount_to_Repay": Total_Amount_to_Repay,
        "duration": duration,
        "New_versus_Repeat": New_versus_Repeat,
        "Amount_Funded_By_Lender": Amount_Funded_By_Lender,
        "Lender_portion_Funded": Lender_portion_Funded,
        "Lender_portion_to_be_repaid": Lender_portion_to_be_repaid,
        "disbursement_date": disbursement_date,
        "due_date": due_date,
        "loan_type": loan_type,
    }
    return data

# Preprocessing function
def preprocess_input(input_data):
    # Extract and transform date features
    disbursement_date = pd.to_datetime(input_data["disbursement_date"])
    due_date = pd.to_datetime(input_data["due_date"])
    disbursement_date_month = disbursement_date.month
    disbursement_date_day = disbursement_date.day
    disbursement_date_year = disbursement_date.year
    due_date_month = due_date.month
    due_date_day = due_date.day
    due_date_year = due_date.year

    # One-hot encode loan type
    loan_type_one_hot = [0] * 24
    loan_type_index = int(input_data["loan_type"].split("_")[1]) - 1
    loan_type_one_hot[loan_type_index] = 1

    # Combine all features
    processed_data = [
        input_data["customer_id"],
        input_data["tbl_loan_id"],
        input_data["lender_id"],
        input_data["Total_Amount"],
        input_data["Total_Amount_to_Repay"],
        input_data["duration"],
        input_data["New_versus_Repeat"],
        input_data["Amount_Funded_By_Lender"],
        input_data["Lender_portion_Funded"],
        input_data["Lender_portion_to_be_repaid"],
        disbursement_date_month,
        disbursement_date_day,
        disbursement_date_year,
        due_date_month,
        due_date_day,
        due_date_year,
        *loan_type_one_hot,
    ]
    
    # Convert to numpy array with correct shape
    return np.array([processed_data], dtype=float)

# Main App
st.title("Phase Group")
st.title("Loan Default Prediction")
st.write("This app predicts loan default risks based on input features.")

input_data = user_input()

if st.button("Predict"):
    try:
        # Preprocess the input data
        feature_array = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(feature_array)
        
        # Display the prediction
        st.write("Prediction:", "Default" if prediction[0] == 1 else "No Default")
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
