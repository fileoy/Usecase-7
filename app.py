import streamlit as st
import requests

# Set the FastAPI endpoint
API_URL = "https://players-prediction.onrender.com/predict"

# Streamlit app title
st.title("Player Performance Prediction")

# Input fields for user data
st.sidebar.header("Input Features")
appearance = st.sidebar.number_input("Appearance", min_value=0, value=10)
assists = st.sidebar.number_input("Assists", min_value=0.0, value=0.0)
days_injured = st.sidebar.number_input("Days Injured", min_value=0, value=10)
games_injured = st.sidebar.number_input("Games Injured", min_value=0, value=2)
award = st.sidebar.number_input("Awards (e.g., MVP Count)", min_value=0, value=0)
highest_value = st.sidebar.number_input("Highest Value (€)", min_value=0, value=100000)

# Create a payload for the API request
payload = {
    "appearance": appearance,
    "assists": assists,
    "days_injured": days_injured,
    "games_injured": games_injured,
    "award": award,
    "highest_value": highest_value,
}

# Predict button
if st.button("Predict"):
    with st.spinner("Predicting..."):
        try:
            # Make a POST request to the FastAPI endpoint
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Raise an error for bad HTTP responses
            prediction = response.json()
            # Display the prediction
            st.success(f"Prediction: {prediction['pred']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
