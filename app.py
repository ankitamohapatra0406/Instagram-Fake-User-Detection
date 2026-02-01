import streamlit as st
import numpy as np
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Instagram Fake Account Detection", layout="centered")
st.title("📸 Instagram Fake vs Real Account Detector")
st.markdown("Enter account details to predict whether the account is **Fake or Genuine**.")
st.divider()

# Input fields
followers = st.number_input("Followers", min_value=0, value=150)
following = st.number_input("Following", min_value=0, value=300)
username_length = st.number_input("Username Length", min_value=1, value=10)
username_has_number = st.selectbox("Username has number?", [0, 1])
full_name_has_number = st.selectbox("Full name has number?", [0, 1])
full_name_length = st.number_input("Full Name Length", min_value=1, value=12)
is_private = st.selectbox("Private Account?", [0, 1])
is_joined_recently = st.selectbox("Joined Recently?", [0, 1])
has_channel = st.selectbox("Has Channel?", [0, 1])
is_business_account = st.selectbox("Business Account?", [0, 1])
has_guides = st.selectbox("Has Guides?", [0, 1])
has_external_url = st.selectbox("Has External URL?", [0, 1])

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[
        followers,
        following,
        username_length,
        username_has_number,
        full_name_has_number,
        full_name_length,
        is_private,
        is_joined_recently,
        has_channel,
        is_business_account,
        has_guides,
        has_external_url
    ]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨Fake Instagram Account Detected")
    else:
        st.success("✅Genuine Instagram Account")
