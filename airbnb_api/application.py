"""Flask App for Predicting Optimal AirBnB prices in Berlin"""
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle
from joblib import load
import xgboost as xgb
import os

# local import:
#from .api_function import get_lemmas
from dotenv import load_dotenv
load_dotenv()

# create app:
def create_app():
    APP = Flask(__name__)

    # load pipeline pickle:
    pipeline1 = load('airbnb_api/test2_regression.pkl')


    @APP.route('/', methods=['POST'])
    def prediction():
        """
        Receives data in JSON format, creates dataframe with data,
        runs through predictive model, returns predicted price as JSON object.
        """

        # Receive data:
        listings = request.get_json(force=True)

        # Features used in predictive model:
        accommodates = listings["accommodates"]
        bathrooms = listings["bathrooms"]
        bedrooms = listings["bedrooms"]
        size = ["size"]
        room_type = listings["room_type"]
        bed_type = listings["bed_type"]
        minimum_nights = listings["minimum_nights"]
        instant_bookable = listings["instant_bookable"]
        cancellation_policy = listings["cancellation_policy"]
        bag_of_words = listings["bag_of_words"]


        features = {
                    'accommodates': accommodates,
                    'bathrooms': bathrooms,
                    'bedrooms': bedrooms,
                    'size': size,
                    'room_type': room_type,
                    'bed_type': bed_type,
                    'minimum_nights': minimum_nights,
                    'instant_bookable': instant_bookable,
                    'cancellation_policy': cancellation_policy,
                    'bag_of_words': bag_of_words
                    }


        # Convert data into DataFrame:
        df = pd.DataFrame([listings], index=[0])
        #df.bag_of_words = get_lemmas(df.bag_of_words.iloc[0])

        # Make prediction for optimal price:
        prediction = pipeline1.predict(df.iloc[0:1])
        output = float(prediction[0])
        print(prediction)

        with open('prediction.txt', 'a') as out:
            out.write(str(output) + '\n')

        with open('prediction.txt', 'r') as out:
            prediction_text = out.readline(1)

        with open('prediction.txt', "rb") as f:
            first = f.readline()        # Read the first line.
            f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
            while f.read(1) != b"\n":   # Until EOL is found...
                f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
            last = f.readline()         # Read last line.


        print(prediction_text)
        final_output = str(last)[2:8]
        #output = pd.DataFrame(output, index=[0]).values.tolist()

        # Return JSON object:
        return jsonify(int(final_output[0:3]))

    return APP
