# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_forecast_predictor_api = Flask("Superkart Sales Forecast Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@sales_forecast_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Forecast Prediction API!"

# Define an endpoint for single property prediction (POST request)
@sales_forecast_predictor_api.post('/v1/predict')
def predict_forecast_price():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product ans store details and returns
    the predicted Product_Store_Sales_Total as a JSON response.
    """
    # Get the JSON data from the request body
    supercart_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': supercart_data['Product_Weight'],
        'Product_Allocated_Area': supercart_data['Product_Allocated_Area'],
        'Product_MRP': supercart_data['Product_MRP'],
        'Store_Age_Years': supercart_data['Store_Age_Years'],
        'Product_Sugar_Content': supercart_data['Product_Sugar_Content'],
        'Product_Type_Category': supercart_data['Product_Type_Category'],
        'Store_Size': supercart_data['Store_Size'],
        'Store_Location_City_Type': supercart_data['Store_Location_City_Type'],
        'Store_Type': supercart_data['Store_Type'],
        'Product_Id_char': supercart_data['Product_Id_char']
    }


    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction 
    predicted_sales  = model.predict(input_data)[0]

     # Convert predicted_price to Python float
    predicted_sales  = round(float(predicted_sales ), 2)
    # The conversion above is needed as we convert the model prediction (log price) to actual price using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted_Product_Store_Sales_Total': predicted_sales })


# Define an endpoint for batch prediction (POST request)
@sales_forecast_predictor_api.post('/v1/predictbatch')
def predict_forecast_price_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing multiple product-store records
    and returns the predicted Product_Store_Sales_Total in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame 
    predicted_sales  = model.predict(input_data).tolist()

    # Convert predicted_price to Python float
    predicted_sales  = [round(float(x), 2) for x in predicted_sales ]
    
    # Add predictions to the DataFrame
    input_data['Predicted_Product_Store_Sales_Total'] = predicted_sales 
    
    # Convert results to dictionary
    output_dict = input_data.to_dict(orient="records")

   # Return the predictions dictionary as a JSON response
 
    return jsonify(output_dict)
    # Return the predictions dictionary as a JSON response
 
# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_forecast_predictor_api.run(debug=True)
