# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 01:14:17 2020

@author: Sukil Siva
"""


from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gevent.pywsgi import WSGIServer


app = Flask(__name__)

classifier = tf.keras.models.load_model("/content/BidirectionalLSTM.h5")

@app.route("/")
def welcome():
  return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict_function():
  if request.method=="POST":
    message = request.form["message"]
    input_data=["message"]
    one_hot_data = [one_hot(words,5000) for words in input_data]
    sequenced_data = pad_sequences(one_hot_data, maxlen=20,padding='pre')
    final_data = np.array(sequenced_data)    
    my_prediction = classifier.predict(final_data)
    return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
  app.run()