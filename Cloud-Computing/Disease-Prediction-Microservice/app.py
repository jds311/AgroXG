from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import requests
import json

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import matplotlib.pyplot as plt
import tensorflow as tf

import math
import random
import datetime


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '../Datav/Model/Wheat_prediction.h5'
print(MODEL_PATH)
# Load your trained model
#print('hi')
model = load_model(MODEL_PATH)
#print('hi3')
#model =tf.keras.models.load_model(MODEL)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
result_final='helllo'

def print_what(classes):
    for i in classes:
        if max(classes[0][0],classes[0][1],classes[0][2],classes[0][3])==classes[0][0]:  return "Crown Root and rot"
        elif max(classes[0][0],classes[0][1],classes[0][2],classes[0][3])==classes[0][1]:  return "Healthy Wheat"
        elif max(classes[0][0],classes[0][1],classes[0][2],classes[0][3])==classes[0][2]:  return "Leaf Rust"
        elif max(classes[0][0],classes[0][1],classes[0][2],classes[0][3])==classes[0][3]:  return "Wheat Loose Smut"

def model_predict(img_path , model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
  #  x = preprocess_input(x, mode='caffe')
    images = np.vstack([x])

    classes = model.predict(images)
    return classes
    
   # python3 -m flask run --host=0.0.0.0 --port=8080

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

@app.route("/temp")
def api():
    with open('../Datav/Data/disease.json') as config_file:
        data = json.load(config_file)
    disease = data['disease']
    filepath = data['filepath']
    # print("diessss "+disease)
    count = data['count']
    count+=1
    data["count"] = count
    with open("../Datav/Data/disease.json", "w") as jsonFile:
        json.dump(data, jsonFile)
    
    response = requests.request("GET","http://172.30.0.3:8082/upload/"+filepath+"/"+disease)
    print(response.text)
    return response.text

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, '../Datav/Upload_Images', secure_filename(f.filename))
        f.save(file_path)
        # file_path1=os.path.join(
        #     basepath, '/home/vatsal/Desktop/AI-CC/AI_Auto_Training/temp_images', secure_filename(f.filename))
        # f.save(file_path1)
        # # Make prediction
        classes = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        result = print_what(classes) # ImageNet Decode
       # result = str(pred_class[0][0][1])               # Convert to string
        result_final = result
        with open("../Datav/Data/disease.json", "r") as jsonFile:
            data = json.load(jsonFile)

        data["disease"] = result
        data["filepath"] = f.filename

        with open("../Datav/Data/disease.json", "w") as jsonFile:
            json.dump(data, jsonFile)

        return result
        
    return None


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8081)
