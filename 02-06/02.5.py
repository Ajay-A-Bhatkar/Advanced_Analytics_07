from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pickle
app = Flask(__name__)

model = joblib.load('Advanced_Analytics_07/02-06/final_salary_model.pkl')
col_names = joblib.load('Advanced_Analytics_07/02-06/salary_column_names.pkl')
@app.route('/')
def hello():
    return "Welcome to the Salary Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON Request
    user_data = request.json

    #convert  JSON  request to DataFrame
    user_df = pd.DataFrame([user_data], columns=col_names)

    # Get prediction
    prediction = model.predict(user_df)

    # Return prediction as JSON
    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
