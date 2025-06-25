import pickle
import json
import numpy as np
import os
# This module is used to load the data and model for house price prediction
__locations = None
__columns_data = None
__model = None

file_dir = os.path.dirname(os.path.abspath(__file__))
columns_file_path = os.path.join(file_dir, 'prices_columns_cp', 'columns_data.json')
model_file_path = os.path.join(file_dir, 'prices_columns_cp','house_price_model.pkl')

def load_data_model():
    print("Loading data and model...")
    #creating global variables for loading data and model
    global __columns_data
    global __locations
    global __model

    with open(columns_file_path, 'r') as f:
        __columns_data = json.load(f)['data_columns']
        print("Columns data loaded successfully.")
        __locations=__columns_data[3:]
        print(__locations)

    with open(model_file_path, 'rb') as f:
        __model = pickle.load(f)
    print("Data and model loaded successfully.")

def get_locations_data():
    return __locations

def get_columns_data():
    return __columns_data
def get_predicted_price(location, bhk, bath, sqft):
    print("Predicting price...")
    try:
        location_index= __columns_data.index(location.lower())
    except:
        location_index= -1
    #creating an empty numpy array with the shape of (1, len(__columns_data))
    x = np.zeros(len(__columns_data))
    #setting the values of the numpy array
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if location_index >= 0:
        x[location_index] = 1
    else:
        x[-1] = 1  # If location not found, set the last index to 1
    #predicting the price
    predicted_price = __model.predict([x])[0]
    print("Price predicted successfully.")
    return round(predicted_price, 2)
load_data_model()
if __name__== "__main__":
    
    get_locations_data()
    print(get_predicted_price("BT", 4 ,3,1100))