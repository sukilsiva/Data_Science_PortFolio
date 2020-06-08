# -*- coding: utf-8 -*-
"""
Created on Sunday May 25 12:53:40 2020

@author: sukil siva.
"""
from flask import Flask, request
from flasgger import Swagger
import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
from glob import glob
import os
import shutil


app = Flask(__name__)
Swagger(app)


### Load the resnet Model thats already been saved
model = tf.keras.models.load_model("resnet-50.model")

@app.route("/")
def Welcome_page():
    return "Welcome All"


@app.route("/get_picture")
def start_web_cam():
    """Lets grab the Details of folders from users
    Before Clicking Images check there is a directory exist 
    ---
    parameters:
        - name: Destination
          in: query
          type: string
          required: true
        
    responses:
        200:
            description: The output values
    """
    image_destination = request.args.get("Destination")
 
    cap = cv2.VideoCapture(0)
    count=0
    while True:
        ret, img = cap.read()
        cv2.imshow("Press Space to Click Images and ESC to Exit", img)
    
        if not ret:
            break
    
        k = cv2.waitKey(1)
    
        if k%256==27:
            # For ESC Key
            print("Program is Closing")
            break
        if k%256==32:
            #For Space Key
            file=image_destination+"img"+str(count)+".jpg"
            cv2.imwrite(file, img)
            count+=1
        
        cap.release()
        cv2.destroyAllWindows()
        
@app.route("/generate_model", methods=["POST"])
def generate_model():
  """Give The Training and Testing Directory
     Before Feeding Images Give a valid directory exist
     ---
     parameters:
         - name : Data_Directory_1
           in : query
           type : string
           required : true
         - name : Label1
           in : query
           type : string
           required : true
         - name : Data_Directory_2
           in : query
           type : string
           required : true
         - name : Label2
           in : query
           type : string
           required : true
         - name : train_folder
           in : query
           type : string
           required : true
         - name : test_folder
           in : query
           type : string
           required : true
         - name : Image_Width
           in : query
           type : number
           required : true
         - name : Image_Height
           in : query
           type : number
           required : true
         - name : Number_of_Epochs
           in : query
           type : number
           required : true
         - name: Image_To_Be_Predicted
           in: formData
           type: file
           required: true
    responses:
        200:
            description: The output values
  """
  
  data_directory_path_case1 = request.args.get("Data_Directory_1")
  data_directory_path_case2 = request.args.get("Data_Directory_2")
  img_width = request.args.get("Image_Width")
  img_height = request.args.get("Image_Height")
  EPOCHS = request.args.get("Number_of_Epochs")
  train = request.args.get("train_folder")
  test = request.args.get("Test_folder")
  case1 = request.args.get("Label1")
  case2 = request.args.get("Label2")
  test_image = request.files.get("Image_To_Be_Predicted")
  IMG_SIZE = [img_width, img_height]
  
  ### Split the Model into training and testing data for case1
  folder = glob(data_directory_path_case1)
  
  ### Now Specify the split size and Usually i prefer 80% train and 20%test
  split_1 = int(0.8 * len(folder))
  split_2 = int(0.9 * len(folder))
  
  ###Now the train and test split
  training_directory_path_case1 = folder[:split_1]
  testing_directory_path_case1 = folder[split_2:]
  
  ############################################################################
  
   ### Split the Model into training and testing data for case2
  folder1 = glob(data_directory_path_case2)
  
  ### Now Specify the split size and Usually i prefer 80% train and 20%test
  split_1 = int(0.8 * len(folder1))
  split_2 = int(0.9 * len(folder1))
  
  ###Now the train and test split
  training_directory_path_case2 = folder1[:split_1]
  testing_directory_path_case2 = folder1[split_2:]
  
  
  ###Now its important to split the train and test directory from the above datas
  directory1 = train
  directory2 = test
  
  path = os.path.join(directory1, "Train")
  path1 = os.path.join(directory2, "Test")
  
  os.mkdir(path)
  os.mkdir(path1) ### Directory has been now created Now lets dump the data
  
  ### Dump the train and test images into a folder
  
  train_dump = directory1+"\Train\img"
  test_dump = directory2+"\Test\img"
  
  def dump_data(data, destination):    
    count = 0
    for i in data:
        while count<len(data):
            img = cv2.imread(i)
            path = destination+str(count)+".jpg"
            cv2.imwrite(path, img)
            count += 1
  
  
  ### Dump the Image Datas to form seperate Train and Test With both the categorical Variables
  dump_data(training_directory_path_case1, train_dump)
  dump_data(testing_directory_path_case1, test_dump)
  dump_data(training_directory_path_case2, train_dump)
  dump_data(testing_directory_path_case2, test_dump)
  
  
  ### Specify the image need to be generated
  traindatagenerator = ImageDataGenerator(rescale=1/255.0,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=False)

  testdatagenerator = ImageDataGenerator(rescale=1/255.0,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=False)


  ###Split the training and testing data
  training_data = traindatagenerator.flow_from_directory(directory=train_dump,
                                                        target_size= IMG_SIZE,
                                                        batch_size=16,
                                                        class_mode="binary")

  testing_data = testdatagenerator.flow_from_directory(directory=test_dump,
                                                        target_size= IMG_SIZE,
                                                        batch_size=16,
                                                        class_mode="binary")



  ###Fit the Model to the generator
  model.fit_generator(generator=training_data,
                              steps_per_epoch=len(training_data),
                              epochs=EPOCHS,
                              validation_data=testing_data,
                              validation_steps=len(testing_data))
  ###Now predict the image 
  image_predictions(test_image)
  
  def image_precitions(pic):
      test_image = image.load_img(pic, target_size=IMG_SIZE)
      
      ### Display the test_image with axis off
      plt.axis('off')
      plt.imshow(test_image)
      
      
      ###Now Convert the images into an array
      test_image = image.img_to_array(test_image)
      test_image = np.expand_dims(test_image, axis=0)
      
      ###and finally predict the values
      result = model.predict(test_image)
      print(result)
      if result[0][0] == 1:
        predictions = case1
      else:
        predictions = case2
      
      ### Finally Print the Output of Predictions
      print("Predictions:",predictions)
    
        
    
if __name__=="__main__":
    app.run()
