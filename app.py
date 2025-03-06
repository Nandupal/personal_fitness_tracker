import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data from the CSV files
calories_df = pd.read_csv('calories.csv')
exercise_df = pd.read_csv('exercise.csv')

# Combine the dataframes if necessary (assuming the columns align)
# Modify the merge or concatenation logic based on how your CSV files are structured
data = pd.merge(calories_df, exercise_df, on='User_ID')  # Assuming 'User_ID' is the common column

# Feature selection (update this according to your actual columns)
X = data[['Age', 'Heart_Rate', 'Weight', 'Duration', 'Body_Temp']]  # Adjust based on available columns
y = data['Calories']  # Target variable

# Scaling the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the RandomForestRegressor model
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Save the model and scaler for future use
joblib.dump(model, 'fitness_calorie_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit app code
st.title("Personal Fitness Tracker")

# Input fields for user data
age = st.number_input("Enter your age:", min_value=0)
heart_rate = st.number_input("Enter your heart rate:")
weight = st.number_input("Enter your weight (kg):")
duration = st.number_input("Enter the duration of exercise (mins):")
body_temp = st.number_input("Enter your body temperature (°C):")

# Predict calories
if st.button("Predict Calories Burned"):
    if age > 0 and heart_rate > 0 and weight > 0 and duration > 0 and body_temp > 0:
        # Prepare input data for prediction
        input_data = np.array([[age, heart_rate, weight, duration, body_temp]])
        input_data_scaled = scaler.transform(input_data)
        
        # Predict calories burned
        predicted_calories = model.predict(input_data_scaled)
        st.write(f"Estimated calories burned: {predicted_calories[0]:.2f} kcal")
    else:
        st.error("Please provide valid input for all fields.")
