import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

# Load the trained model
# Assuming the model is saved in the model_building directory
model_path = 'model_building/best_xgboost_model.pkl'

# Download and load the model
model_path = hf_hub_download(repo_id="rekhchan/MLoPs", filename="best_xgboost_model.joblib")
model = joblib.load(model_path)

# Ensure the model path is correct based on where it was saved
# For simplicity, assuming it's best_xgboost_model.pkl as done in similar projects
# If the model is not yet saved, this part will need adjustment once model saving is implemented

# Placeholder for model loading if it's not yet explicitly saved
# For now, we'll assume a model file named 'best_xgboost_model.pkl' exists
# If a model wasn't explicitly saved earlier, a placeholder joblib.load would fail.
# The previous cell only logs model artifacts, not saves a local .pkl.
# For demonstration, let's create a dummy model save here if it's missing.

# To prevent an error if model was not saved by joblib.dump
# We will temporarily add a save command to the model training cell later if needed.

# As per the problem description, the model is saved locally using joblib.
# So, loading should work assuming the save command is present or will be added.

# --- Streamlit App --- #
st.set_page_config(page_title="Wellness Tourism Package Purchase Prediction")
st.title("Wellness Tourism Package Purchase Prediction")

st.markdown("Predict whether a customer will purchase the Wellness Tourism Package based on their details.")

# --- Input Features --- #
st.header("Customer Details")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 35)
    monthly_income = st.number_input("Monthly Income", 10000, 100000, 25000)
    number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 10, 2)
    number_of_trips = st.slider("Number of Trips Annually", 0, 20, 2)
    number_of_children_visiting = st.slider("Number of Children Visiting", 0, 5, 0)

with col2:
    typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    maritalstatus = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col3:
    owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    number_of_followups = st.slider("Number of Follow-ups", 1, 6, 3)
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Premium'])
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])

# --- Prediction Button --- #
if st.button("Predict Purchase"):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'NumberOfTrips': [number_of_trips],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'PitchSatisfactionScore': [pitchsatisfactionscore],
        'NumberOfFollowups': [number_of_followups],
        'DurationOfPitch': [duration_of_pitch],
        'Passport': [passport],
        'OwnCar': [owncar],
        'TypeofContact_Company Invited': [1 if typeofcontact == 'Company Invited' else 0],
        'Occupation_Large Business': [1 if occupation == 'Large Business' else 0],
        'Occupation_Salaried': [1 if occupation == 'Salaried' else 0],
        'Occupation_Small Business': [1 if occupation == 'Small Business' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'MaritalStatus_Married': [1 if maritalstatus == 'Married' else 0],
        'MaritalStatus_Single': [1 if maritalstatus == 'Single' else 0],
        'Designation_Manager': [1 if designation == 'Manager' else 0],
        'Designation_Senior Manager': [1 if designation == 'Senior Manager' else 0],
        'Designation_Executive': [1 if designation == 'Executive' else 0],
        'Designation_AVP': [1 if designation == 'AVP' else 0],
        'Designation_VP': [1 if designation == 'VP' else 0],
        'Designation_Director': [1 if designation == 'Director' else 0],
        'ProductPitched_Deluxe': [1 if product_pitched == 'Deluxe' else 0],
        'ProductPitched_King': [1 if product_pitched == 'King' else 0],
        'ProductPitched_Premium': [1 if product_pitched == 'Premium' else 0],
        'ProductPitched_Standard': [1 if product_pitched == 'Standard' else 0],
        'ProductPitched_Super Deluxe': [1 if product_pitched == 'Super Deluxe' else 0],
        'CityTier_2': [1 if city_tier == 2 else 0],
        'CityTier_3': [1 if city_tier == 3 else 0],
        'PreferredPropertyStar_4': [1 if preferred_property_star == 4 else 0],
        'PreferredPropertyStar_5': [1 if preferred_property_star == 5 else 0]
    })

    # Ensure all columns expected by the model are present and in the correct order
    # This assumes X_train.columns has been saved or can be reconstructed
    # For this example, let's explicitly list columns based on the notebook's preprocessing
    # and fill missing with 0 for product_pitched, designation, etc.

    # Reconstruct the column list based on the preprocessing steps in the notebook
    # from cell ZavGVd_5Manu (X = df.drop('ProdTaken', axis=1)) and the one-hot encoding.
    # Ensure proper ordering and all dummy variables are handled.
    expected_columns = [
        'Age', 'MonthlyIncome', 'NumberOfPersonVisiting', 'NumberOfTrips',
        'NumberOfChildrenVisiting', 'PitchSatisfactionScore', 'NumberOfFollowups',
        'DurationOfPitch', 'Passport', 'OwnCar',
        'TypeofContact_Company Invited',
        'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business',
        'Gender_Male',
        'MaritalStatus_Married', 'MaritalStatus_Single',
        'Designation_AVP', 'Designation_Director', 'Designation_Executive', 'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP',
        'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Premium', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
        'CityTier_2', 'CityTier_3',
        'PreferredPropertyStar_4', 'PreferredPropertyStar_5'
    ]

    # Align input_data with expected_columns. Fill missing with 0.
    # This also handles cases where a dummy variable might not be created for certain inputs (e.g., if only 'Male' is chosen for gender).
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_columns]

    try:
        # Load the model
        model = joblib.load(model_path)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        # Use the classification threshold used during training (0.45)
        classification_threshold = 0.45
        prediction = (prediction_proba >= classification_threshold).astype(int)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"The customer is predicted to PURCHASE the Wellness Tourism Package with a probability of {prediction_proba[0]:.2f}.")
        else:
            st.info(f"The customer is predicted NOT to purchase the Wellness Tourism Package with a probability of {prediction_proba[0]:.2f}.")
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
