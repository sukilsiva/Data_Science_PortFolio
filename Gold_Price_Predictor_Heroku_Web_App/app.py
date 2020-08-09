# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 00:59:42 2020

@author: Sukil Siva
"""


### Import the Libraries
import pickle 
import numpy as np
from flask import Flask, request, render_template

### Get the Pickle Model From Local Disk
pickle_in = open("model.pkl","rb")
regressor=pickle.load(pickle_in)

### Load the MinMaxScaler
scaler = pickle.load(open("Scaler.pkl", "rb"))

### Start the App
app=Flask(__name__)

### Main Page of the Web Page
@app.route('/')
def welcome():
    return render_template("index.html")


### Get inputs from the user    
@app.route('/predict', methods=["POST"])
def predict():
    if request.method=="POST":
        ### Collecting the Information given by user using request lib
        day1 = int(request.form['Day1'])
        day2 = int(request.form['Day2'])
        day3 = int(request.form['Day3'])
        day4 = int(request.form['Day4'])
        day5 = int(request.form['Day5'])
        
        ### Scaling the data and transforming into 1 rows and 5 Features
        data=scaler.fit_transform(np.array([day1,day2,day3,day4,day5]).reshape(-1,1))
        data=data.reshape(1,5)
        
        ### Prediction
        my_prediction = regressor.predict(data)
        
        my_prediction=scaler.inverse_transform(my_prediction.reshape(-1,1))
        
        my_prediction=np.round(my_prediction,decimals=2)
        
        my_prediction=my_prediction[0]
        
        return render_template('result.html', prediction='The Rate of Gold in INR will be {}'.format(my_prediction))

if __name__=="__main__":
    app.run()

