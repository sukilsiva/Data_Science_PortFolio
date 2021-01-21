# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:12:02 2020

@author: Sukil Siva
"""

### Importing the Libraries
from flask import Flask, request, render_template
import numpy as np
import pickle
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io 
from flaskext.mysql import MySQL

### Get the Pickle Model From Local Disk
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

### Load the MinMaxScaler
scaler = pickle.load(open("scaler.pkl", "rb"))

### Starting the App
app = Flask(__name__)

### COnfigurations are been set up for DB and Flask App
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'Sathya@1972'
app.config['MYSQL_DATABASE_DB'] = 'churnapp'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

### Initiating the Flask app
mysql = MySQL()
mysql.init_app(app)

@app.route('/')
def welcome():
    return render_template("index.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method=="POST":
        ### Collecting Information using request Library

        customerid = str(request.form["CustomerID"])

        Gender = "Male"
        if request.form["gender"] == 1:
            Gender = "Female"

        seniorcitizen = 0
        if 'SeniorCitizen' in request.form:
            seniorcitizen = 1

        partner = "No"
        if 'partner' in request.form:
            partner = "Yes"    
    
    
        dependents = 0
        if 'Dependents' in request.form:
            Dependents = 1
        
        phoneservice = "No"
        if 'PhoneService' in request.form:
            phoneservice = "Yes"

        multiplelines = "No"
        if 'MultipleLines' in request.form:
            multiplelines = "Yes"
        
        billing = "No"
        if 'PaperlessBilling' in request.form:
            billing = "Yes"
        
        onlinebackup = "No"
        if 'OnlineBackup' in request.form:
            onlinebackup = "Yes"
        
        deviceprotection = "No"
        if 'DeviceProtection' in request.form:
            deviceprotection = "Yes"
        
        streamingTV = "No"
        if 'StreamingTV' in request.form:
            streamingTV = "Yes"
        
        streamingMovies = "No"
        if 'StreamingMovies' in request.form:
            streamingMovies = "Yes"
        
        InternetService_No = "No"
        if request.form["InternetService"] == 1:
            InternetService_No = "DSL"
        elif request.form["InternetService"] == 2:
            InternetService_No = "Fiber Optic"

        paymentmethod = "Bank Transfer (Automatic)"
        if request.form["PaymentMethod"] == 1:
            paymentmethod = "Credit Card (Automatic)"
        elif request.form["PaymentMethod"] == 2:
            paymentmethod = "Electronic Check"
        elif request.form["PaymentMethod"] == 3:
            paymentmethod = "Mailed Check"
        
        
        onlinesecurity = 0
        if 'OnlineSecurity' in request.form and InternetService_No == 2:
            onlinesecurity = 1
        
        techsupport = 0
        if 'TechSupport' in request.form and InternetService_No == 2:
            techsupport = 1
            
        contract = 0
        if request.form["Contract"] == 1:
            contract = 1
        elif request.form["Contract"] == 2:
            contract = 2
        
        monthlyCharges = int(request.form["MonthlyCharges"])
        Tenure = int(request.form["Tenure"])

        totalCharges = str(monthlyCharges * Tenure)
        
        ### Fitting the Data for Scaling the Values
        data=scaler.fit_transform(np.array([contract, onlinesecurity, techsupport, Tenure, monthlyCharges, seniorcitizen, dependents]).reshape(-1,1))
       
        ### Prediction
        prediction=classifier.predict(np.reshape(data, (1,7)))
        
        ###Prediction Probability
        prediction_proba = classifier.predict_proba(np.reshape(data, (1,7)))
        
        my_prediction = prediction[0]
        my_prediction_proba = np.round(prediction_proba[0,1], 2)
        
        if  my_prediction == 0:
            churn = "No"
        else:
            churn = "yes"

        if seniorcitizen == 0:
            seniorcitizen = "No"
        else:
            seniorcitizen = "Yes"
        
        if dependents == 0:
            dependents = "No"
        else:
            dependents = "Yes"
        
        if techsupport == 0:
            techsupport = "No"
        else:
            techsupport = "Yes"
        
        if onlinesecurity == 0:
            onlinesecurity= "No"
        else:
            onlinesecurity = "Yes"
        
        if contract == 0:
            contract = "Month-to-Month"
        elif contract == 1:
            contract = "One-Year"
        elif contract == 2:
            contract = "Two-Year"
        
        
        query = "INSERT INTO webappdata (CustomerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (customerid, Gender, seniorcitizen, partner, dependents, Tenure, phoneservice, multiplelines, InternetService_No, onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingTV, streamingMovies, contract, billing, paymentmethod, monthlyCharges, totalCharges, churn)
        conn = mysql.connect()
        cur = conn.cursor()
        cur.execute(query, values)
        conn.commit()
        cur.close()
    
        return render_template("index.html",prediction_text='Churn probability is {} and the Churn is {}'.format(my_prediction_proba, churn))


if __name__ == "__main__":
    app.run(port = 8080, debug =True)


