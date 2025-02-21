from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open("xgboost_ppd_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Postpartum Depression Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Get probability of positive class
        
        return jsonify({"prediction": int(prediction), "probability": float(probability)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
