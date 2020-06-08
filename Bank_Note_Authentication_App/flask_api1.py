# -*- coding: utf-8 -*-
"""
Created on Mon May 18 01:40:38 2020

@author: sukil siva
"""


from flask import Flask, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome Everyone to my app"

@app.route('/predict', methods = ["Get"])
def bank_note_authentication():
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = classifier.preict(([variance, skewness, curtosis, entropy]))
    return "The answer is" + str(prediction)

app.route('/predict_file', methods = ["POST"])
def note_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return str(list(prediction))


if __name__ == '__main__':
    app.run()