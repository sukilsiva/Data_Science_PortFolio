from locust import HttpUser, task, between                    # The Package im going to use for Loadtesting is Locust
import json                                                   #  Since the Incoming data isin Jsocls format we are imoporting the json library

class LoadTesting(HttpUser):
    wait_time = between(1,2)                                  # It provides the User shoud wait for 1-2 secs for every input

    @task(1)
    def testing_the_app(self):
        #payload = {"MonthlyCharges" : "1000", "Tenure" : "12"}       # Payload we need to specify the nput to application
        #header = {"Content-Type" : "application/json", "Accept" : "application/json"}# Accepts inputs from user in json format in Application
        self.client.post("/predict", data={"CustomerID": "1235-ABDE","gender": "0","SeniorCitizen": "0","Partner": "0","Dependents": "0","PhoneService": "0","MultipleLines": "0","PaperlessBilling": "0","OnlineBackup": "0","DeviceProtection":"0","StreamingTV":"0","StreamingMovies":"0","InternetService": "1","PaymentMethod":"1","OnlineSecurity":"0","TechSupport":"0", "Contract": "1", "MonthlyCharges" : "1000", "Tenure" : "12"},catch_response = False) # Post the Comments to App

        