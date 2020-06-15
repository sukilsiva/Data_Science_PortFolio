# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:10:53 2020

@author: Sukil Siva
"""
### import the Libraries

import pickle 
from flask import Flask, render_template, request

### Dump the Model
pickle_in = open("Classifier.pkl","rb")
classifier = pickle.load(pickle_in)
cv = pickle.load(open("CVtransform.pkl", "rb"))

### Start the app
app = Flask(__name__)

### Defining the Welcome Page
@app.route('/')
def welcome():
    return render_template("index.html")


### Get inputs from the user    
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)



if __name__ =="__main__":
    app.run()


