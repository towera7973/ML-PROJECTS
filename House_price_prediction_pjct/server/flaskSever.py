from flask import Flask, request, jsonify
import utility
import numpy as np
# Importing the utility module which contains the get_locations_data function
app_obj = Flask(__name__)

@app_obj.route("/")
def index():
    return "Welcome to the House Price Prediction API! Use /get_loc_data or /predict_home_price."
@app_obj.route("/get_loc_data", methods=['GET'])
def get_loc_data():
    loc_response= jsonify(
        {"locations":utility.get_locations_data()}
        )
    loc_response.headers.add('Access-Control-Allow-Origin', '*')
    return loc_response

@app_obj.route("/predict_home_price", methods=['POST'])
def predict_home_price():
    location = request.form["location"]
    total_sqft= float(request.form["total_sqft"])
    bath= int(request.form["bath"])
    bedroom=int( request.form["bedroom"])

    response =jsonify(
        { "predicted_price": utility.get_predicted_price(location, total_sqft, bath, bedroom) }
    )
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    utility.load_data_model()
    app_obj.run(debug=True)