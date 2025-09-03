import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# --- تحميل وإعداد البيانات والنماذج (يتم مرة واحدة) ---
@st.cache_data
def load_data_and_preprocessors():
    """Load training data and prepare processing tools (Scaler and Encoder)"""
    train_df = pd.read_csv(os.path.join("data", "train.csv"))
    
    # Prepare and train LabelEncoder for sex
    le = LabelEncoder()
    le.fit(train_df['Sex'])
    
    # Prepare and train MinMaxScaler for numerical features
    scaler = MinMaxScaler()
    numerical_cols = train_df.drop(columns=['id', 'Calories', 'Sex']).columns
    scaler.fit(train_df[numerical_cols])
    
    return le, scaler, numerical_cols

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = os.path.join("model", "XGBRegressor_best_model.pkl")
    model = joblib.load(model_path)
    return model

# Prepare the app page
st.set_page_config(page_title="Calories Prediction", page_icon="🔥", layout="centered")
st.title("🔥 Calories Prediction App")
st.markdown("Enter your data to predict the amount of calories burned.")

# --- Load resources ---
try:
    model = load_model()
    le, scaler, numerical_cols = load_data_and_preprocessors()
except FileNotFoundError as e:
    st.error(f"❌ A necessary file was not found: {e}. Make sure the files are in the correct paths.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading resources: {e}")
    st.stop()

# --- Data input interface ---
st.subheader("🔢 Enter your workout data:")

# Create columns to organize the interface
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

with col2:
    duration = st.number_input("Workout Duration (minutes)", min_value=1, max_value=300, value=30)
    heart_rate = st.number_input("Average Heart Rate", min_value=50, max_value=220, value=120)
    body_temp = st.number_input("Body Temperature (Celsius)", min_value=35.0, max_value=45.0, value=38.0, format="%.1f")

# --- Prediction and display results ---
if st.button("🔍 Calculate Calories"):
    try:
        # 1. Create DataFrame from user inputs
        input_data = pd.DataFrame([{
            'Sex': sex,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Duration': duration,
            'Heart_Rate': heart_rate,
            'Body_Temp': body_temp
        }])

        # 2. Apply preprocessing using loaded tools
        # Convert sex
        input_data['Sex'] = le.transform(input_data['Sex'])

        # Normalize numerical features
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Ensure column order is the same as the model
        final_input = input_data[model.feature_names_in_]

        # 3. Make prediction
        prediction = model.predict(final_input)[0]

        st.success(f"🔥 Predicted Calories: {prediction:.2f} kcal")
    except Exception as e:
        st.error(f"An error occurred during the prediction process: {e}")