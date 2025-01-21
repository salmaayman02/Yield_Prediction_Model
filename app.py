import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pickle

# File paths
model_file = "final_model.h5"
area_encoder_file = "label_encoder_area.pkl"
item_encoder_file = "label_encoder_item.pkl"
scaler_file = "scaler_features.pkl"
target_scaler_file = "scaler_target.pkl"
dataset_file = "yield_df.csv"

# Load the trained model
final_model = load_model(model_file, custom_objects={"mse": MeanSquaredError()})

# Load saved preprocessing artifacts
with open(area_encoder_file, 'rb') as f:
    area_encoder = pickle.load(f)
with open(item_encoder_file, 'rb') as f:
    item_encoder = pickle.load(f)
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)
with open(target_scaler_file, 'rb') as f:
    target_scaler = pickle.load(f)

# Load the dataset to extract options for categorical input fields
df = pd.read_csv(dataset_file)
areas = df['Area'].unique().tolist()
items = df['Item'].unique().tolist()

# Streamlit App
st.title("Yield Prediction App")
st.write("Enter the feature values below to predict the yield:")

# Input fields
area = st.selectbox("Select Area", options=areas)
item = st.selectbox("Select Item", options=items)
average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm per year)", min_value=0.0, step=0.1)
pesticides_tonnes = st.number_input("Pesticides (tonnes)", min_value=0.0, step=0.1)
avg_temp = st.number_input("Average Temperature (°C)", min_value=-.0, max_value=50.0, step=0.1)
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)

if st.button("Predict Yield"):
    try:
        # Encode categorical inputs
        area_encoded = area_encoder.transform([area])[0]
        item_encoded = item_encoder.transform([item])[0]

        # Prepare and scale numerical features
        numerical_features = np.array([
            float(average_rain_fall_mm_per_year),
            float(pesticides_tonnes),
            float(avg_temp),
            int(year)
        ]).reshape(1, -1)

        scaled_numerical_features = scaler.transform(numerical_features)

        # Combine encoded categorical and scaled numerical features
        input_array = np.hstack(([[area_encoded, item_encoded]], scaled_numerical_features))


        # # Reverse scaling for target variable
        predicted_yield = target_scaler.inverse_transform(scaled_prediction)[0][0]
        
        

        
        # Display results
        st.success(f"Predicted Yield: {predicted_yield:.2f} hg/ha")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
