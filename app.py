import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb  # Ensure XGBoost is installed

# Load the model
@st.cache_resource
def load_model():
    model_path = "xgboost_model.pkl"
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Load test dataset
@st.cache_data
def load_data():
    test_data_path = "Test.csv"
    return pd.read_csv(test_data_path)

# Preprocess input
def preprocess_input(data, columns):
    """Ensure data matches the model's expected format"""
    return data[columns]

# Streamlit UI
st.title("Loan Prediction App")
st.write("Predict loan outcomes using the provided model and dataset.")

# Upload file option
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
if uploaded_file:
    user_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(user_data)

    # Preprocess and predict
    model = load_model()
    try:
        predictions = model.predict(user_data)  # Ensure `user_data` matches expected input
        user_data["Prediction"] = predictions
        st.write("Prediction Results:")
        st.dataframe(user_data)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Test prediction
st.write("Or use sample data:")
sample_data = load_data()
if st.button("Run Predictions on Sample Data"):
    model = load_model()
    try:
        sample_predictions = model.predict(sample_data)
        sample_data["Prediction"] = sample_predictions
        st.write("Sample Data with Predictions:")
        st.dataframe(sample_data)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
