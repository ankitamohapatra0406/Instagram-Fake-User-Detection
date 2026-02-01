import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
model = joblib.load(MODEL_PATH)

# Example input
sample_input = np.array([[
    150,   # followers
    300,   # following
    10,    # username_length
    1,     # username_has_number
    0,     # full_name_has_number
    12,    # full_name_length
    0,     # is_private
    1,     # is_joined_recently
    0,     # has_channel
    0,     # is_business_account
    0,     # has_guides
    1      # has_external_url
]])
prediction = model.predict(sample_input)

if prediction[0] == 1:
    print("Fake Instagram Account")
else:
    print("Genuine Instagram Account")
