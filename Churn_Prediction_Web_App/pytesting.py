import os
import requests
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sqlalchemy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict 
from imblearn.combine import SMOTETomek
import json 

class testing():
    def __init__(self, original_data, webappdata):
        self.availdata = original_data
        self.webappdata = webappdata
    def get_data(self): 
        # data from MySQL Database
        #https://dev.to/fpim/object-oriented-design-architecture-with-panda-me4     ### Reference for data cleaning and making perfect data
        
        ### Have an Connection to Database
        engine = sqlalchemy.create_engine('mysql+pymysql://root:idntknwpassword@localhost:3306/churnapp')
        avail_df = pd.read_sql_table(self.availdata, engine)
        app_df = pd.read_sql_table(self.webappdata, engine)
        
        raw_data = pd.concat([avail_df, app_df], axis=0)

        return raw_data
    
    def preprocess_input(self):
        
        raw_data = self.get_data()

        df = pd.read_csv(raw_data)

        all_features = df.columns
        
        ### Make the Total Charges to a Numeric Value
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], downcast='integer', errors='coerce')

        ### Since the Data is Skewed
        numerical_feature = [feature for feature in df.columns if df[feature].dtypes!="O"]
        numerical_feature.remove('SeniorCitizen')
        for feature in numerical_feature:
            df[feature] = np.log(df[feature])

        ### Picking up Categorical Variables
        categoircal_features = [feature for feature in all_features if feature not in numerical_feature + 'customerID' + 'Churn']
        
        ### Handling Rare Categorical Variables
        categoircal_features.remove("TotalCharges")
        for feature in categoircal_features:
            data = df.copy()
            data["Churn"] = np.where((data["Churn"]=="Yes"),1,0)
            temp = df.groupby(feature)["Churn"].count()/len(data)                              #Collecting the Total Values
            temp_df = temp[temp>0.01].index                                                  # Values Greater that .01% index are Noted
            df[feature] = np.where(df[feature].isin(temp_df), df[feature], "Rare_var") # Replacing rare Variables
        
        
        ### Transforming the Categorical Variables into Labels
        # Lets do with Label Encoding since there is no Cardinality
        categoircal_features.append("Churn")
        data = df[categoircal_features]                                                                   
        df.drop(categoircal_features, axis=1, inplace=True)                                 #Dropping Categorical Vairables
        data = data.apply(LabelEncoder().fit_transform)                                         # Transforming the characters to Labels
        
        ### Concating the Whole Dataset into a Single Data
        df = pd.concat([df,data], axis=1)
        
        ### Scaling the Value 
        scaling_feature = [feature for feature in all_features if feature not in ["customerID", "Churn"]]

        # Since the Columns Tenure has infinte Values in it
        df.replace([np.inf, -np.inf], 0, inplace=True)       # Replacing inf values with 0

        ### fitting the values to Scaler function
        scaler=MinMaxScaler()
        scaler.fit(df[scaling_feature])
        df = pd.concat([df[["customerID", "Churn"]].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(df[scaling_feature]), columns=scaling_feature)],
                    axis=1)
        ### Arranging the Columns with respet to dataset
        df=df[[feature for feature in data.columns]]

        return df 

    def data_testing(self):

        dataframe = self.preprocess_input()

        # read the processed data
        df = pd.read_csv(dataframe)

        # Taking only the selected Features
        selected_features = ["Contract", "OnlineSecurity", "TechSupport", "tenure", "MonthlyCharges", "SeniorCitizen", "Dependents", "Churn"]
        df = df[selected_features]

        # Dependent and Independent Variables Split up
        X = df.drop("Churn", axis=1)
        Y = df["Churn"]

        # RandomForestClassifier Model
        clf = RandomForestClassifier()
        yhat = cross_val_predict(clf, X, Y, cv=5)

        acc = np.mean(yhat== Y)
        tn, fp, fn, tp = confusion_matrix(Y, yhat).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp + fn)

         # Let's visualize within several slices of the dataset
        score = yhat == Y 
        score_int = [int(s) for s in score]
        df['pred_accuracy'] = score_int

        # Bar plot by region

        sns.set_color_codes("dark")
        ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette = "Greens_d")
        ax.set(xlabel="Region", ylabel = "Model accuracy")
        figure = plt.savefig("by_region.png",dpi=80)

        # Now print to file
        with open("metrics.json", 'w') as outfile:
            json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

        return outfile, figure

       


