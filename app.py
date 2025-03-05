from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('calories_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        # Create a feature array in the same order as the model's features
        user_data = np.array([[age, height, weight, duration, heart_rate, body_temp]])

        # Scale the user input data using the saved scaler
        user_data_scaled = scaler.transform(user_data)

        # Predict the calories using the loaded model
        calories_burned = model.predict(user_data_scaled)[0]

        # Render the result back to the webpage
        return render_template('fitness_data.html', prediction=calories_burned)

if __name__ == '__main__':
    app.run(debug=True)
