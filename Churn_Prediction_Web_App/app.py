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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import io

### Get the Pickle Model From Local Disk
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

### Load the MinMaxScaler
scaler = pickle.load(open("scaler.pkl", "rb"))

### Starting the App
app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template("index.html")

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method=="POST":
        ### Collecting Information using request Library
        SeniorCitizen = 0
        if 'SeniorCitizen' in request.form:
            SeniorCitizen = 1
        
        Dependents = 0
        if 'Dependents' in request.form:
            Dependents = 1
        
        InternetService_No = 0
        if request.form["InternetService"] == 1:
            InternetService_No = 1
        elif request.form["InternetService"] == 2:
            InternetService_No = 2
            
        OnlineSecurity = 0
        if 'OnlineSecurity' in request.form and InternetService_No == 2:
            OnlineSecurity = 1
        
        TechSupport = 0
        if 'TechSupport' in request.form and InternetService_No == 2:
            TechSupport = 1
            
        Contract = 0
        if request.form["Contract"] == 1:
            Contract = 1
        elif request.form["Contract"] == 2:
            Contract = 2
        
        MonthlyCharges = int(request.form["MonthlyCharges"])
        Tenure = int(request.form["Tenure"])
        
        ### Fitting the Data for Scaling the Values
        data=scaler.fit_transform(np.array([Contract, OnlineSecurity, TechSupport, Tenure, MonthlyCharges, SeniorCitizen, Dependents]).reshape(-1,1))
        
        ### printing the Data
        print(data)
        
        ### Prediction
        prediction=classifier.predict(np.reshape(data, (1,7)))
        
        ###Prediction Probability
        prediction_proba = classifier.predict_proba(np.reshape(data, (1,7)))
        
        my_prediction = prediction[0]
        my_prediction_proba = np.round(prediction_proba[0,1], 2)
        
        
        ### Gauge Plot
        
        def degree_range(n):
            start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
            end = np.linspace(0,180,n+1, endpoint=True)[1::]
            mid_points = start + ((end-start)/2.)
            return np.c_[start, end], mid_points

        def rot_text(ang):
            rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
            return rotation
    
        def gauge(labels=['LOW','MEDIUM','HIGH','EXTREME'], \
                  colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], Probability=1, fname=False):
    
            N = len(labels)
            colors = colors[::-1]
    
    
            """
            begins the plotting
            """
    
            gauge_img = io.BytesIO()
            fig, ax = plt.subplots()
    
            ang_range, mid_points = degree_range(4)
    
            labels = labels[::-1]
    
            """
            plots the sectors and the arcs
            """
            patches = []
            for ang, c in zip(ang_range, colors):
                # sectors
                patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
                # arcs
                patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
    
            [ax.add_patch(p) for p in patches]
    
    
            """
            set the labels (e.g. 'LOW','MEDIUM',...)
            """
    
            for mid, lab in zip(mid_points, labels):
    
                ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                    horizontalalignment='center', verticalalignment='center', fontsize=14, \
                    fontweight='bold', rotation = rot_text(mid))
    
            """
            set the bottom banner and the title
            """
            r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
            ax.add_patch(r)
    
            ax.text(0, -0.05, 'Churn Probability ' + Probability.astype(str), horizontalalignment='center', \
                 verticalalignment='center', fontsize=22, fontweight='bold')
    
            """
            plots the arrow now
            """
    
            pos = (1-Probability)*180
            ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                         width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
            ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
            ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))
    
            """
            removes frame and ticks, and makes axis equal and tight
            """
    
            ax.set_frame_on(False)
            ax.axes.set_xticks([])
            ax.axes.set_yticks([])
            ax.axis('equal')
            plt.tight_layout()
    
            plt.savefig(gauge_img, format = 'png')
            gauge_img.seek(0)
            url = base64.b64encode(gauge_img.getvalue()).decode()
            return url
        
        gauge_url = gauge(Probability=my_prediction_proba)
        
        if  my_prediction == 0:
            prediction_test = "No"
        else:
            prediction_test = "yes"
            
        return render_template("results.html",prediction_text='Churn probability is {} and the Churn is {}'.format(my_prediction_proba, prediction_test), url1 = gauge_url)


if __name__ == "__main__":
    app.run(port = 8080)


