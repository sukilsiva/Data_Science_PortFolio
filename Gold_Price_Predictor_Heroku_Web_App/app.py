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
scaler = pickle.load(open("scaler.pkl", "rb"))

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
        
        
        ###Prediction for the Next 1 Week
        list_output=[]
        for i in range(0,5):
            prediction=regressor.predict(np.reshape(data,(1,5)))
            list_output.append(prediction[0])
            data=data[1:]
            data=np.append(data, prediction[0])
            i+=1
        
        ###Inverse Transfrom of scaled_output
        list_output = np.array(list_output)
        my_prediction=scaler.inverse_transform(list_output.reshape(-1,1))
        
        ###Round the values to Two Integers
        my_prediction=np.round(my_prediction,2)
        
        return render_template('result.html', prediction="Monday:Rs.{}  \n Tuesday:Rs.{}  \n  Wednesday:Rs.{}  \n  Thursday:Rs.{}  \n   Friday:Rs.{}".format(my_prediction[0][0],my_prediction[1][0],my_prediction[2][0],my_prediction[3][0],my_prediction[4][0]))

if __name__=="__main__":
    app.run()
