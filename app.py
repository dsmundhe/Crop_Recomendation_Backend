from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd  # ✅ Added to avoid warnings

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load the trained model
model = joblib.load('crop_recommendation_model.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request

    # Ensure the data contains all necessary features
    try:
        feature_values = [
            data['N'], 
            data['P'], 
            data['K'], 
            data['temperature'], 
            data['humidity'], 
            data['ph'], 
            data['rainfall']
        ]
    except KeyError:
        return jsonify({'error': 'Missing required features'}), 400

    # Create DataFrame with correct feature names (to avoid sklearn warning)
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    features = pd.DataFrame([feature_values], columns=columns)

    # Make prediction using the model
    prediction = model.predict(features)

    # Return the prediction result as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
