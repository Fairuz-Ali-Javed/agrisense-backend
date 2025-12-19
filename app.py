import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()
pkl1 = os.getenv("PKL1")
pkl2 = os.getenv("PKL2")
pkl3 = os.getenv("PKL3")

AgriSense = Flask(__name__)
CORS(AgriSense)

clf = joblib.load(pkl1)
ct = joblib.load(pkl3)
sc = joblib.load(pkl2)

#===========================================================================================
#the feature names in the same order as in original training data or my matrix of features.
# take input values and convert them into a dataframe and then use the "clf" to predict
# which crop to grow as a list and then take that crop and return it as a json file
#============================================================================================
inputs = ["Crop_Year", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]

@AgriSense.route('/prediction', methods=['POST'])
def crop_predict():     
    data = request.get_json(force=True)

    df = pd.DataFrame([data], columns=inputs)
    processed = ct.transform(df)
    scaled = sc.transform(processed)

    recommended_crop = clf.predict(scaled)[0]

    return jsonify({'predicted_crop': recommended_crop})

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    AgriSense.run(debug=True, port=5000)