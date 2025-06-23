import pickle
import json
import numpy as np

def load_data_model():
    print("Loading data and model...")
    #creating global variables for loading data and model
    global __columns_data
    global __locations
    global __model

    with open('server/columns_data.json', 'r') as f:
        __columns_data = json.load(f)
        __locations=__columns_data[3:]
    with open('server/model.pkl', 'rb') as f:
        __model = pickle.load(f)
    print("Data and model loaded successfully.")

    