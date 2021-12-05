from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.applications import EfficientNetB0
from imutils import paths
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
import logging

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # logging.basicConfig(filename='/home/vatsal/Desktop/AI-CC/AI_Auto_Training/log_files/train.log',level=logging.DEBUG)
    print('libraries imported')
    #logging.basicConfig(filename='train.log',level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(messege)s')
    logging.debug('Libraries Imported')

    #     ------------------------- Data Preprocessing -------------------------
    LABELS = set(["Crown and Root Rot", "Healthy Wheat", "Leaf Rust", "Wheat Loose Smut"])
    #imagePaths = list(paths.list_images('./home/vatsal/Desktop/AI-CC/AI_Auto_Training/Large Wheat Disease Classification Dataset'))
    imagePaths = list(paths.list_images('../Datav/Large Wheat Disease Classification Dataset'))
    data = []
    labels = []
    #print(imagePaths)
    # loop over the image paths
    for imagePath in imagePaths:
    # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
    # if the label of the current image is not part of the labels
    # are interested in, then ignore the image
        if label not in LABELS:
            continue
    # load the image, convert it to RGB channel ordering, and resize
    # it to be a fixed 224x224 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
    # print('helllo')
    print('Preprocessing Done')


    #     ------------------------- Data Labelling and Splitting the Data -------------------------
    # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    #print(labels)
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, stratify=labels, random_state=42) 
    print('Data Labelling and Splitting the Data Done')

    #     ------------------------- Data Augmentation -------------------------
    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
    # initialize the validation/testing data augmentation object (which
    # we'll be adding mean subtraction to)
    valAug = ImageDataGenerator()
    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation
    # objects
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    trainAug.mean = mean
    valAug.mean = mean
    print(' Data Augmentation Done')

    #   ------------------------- Model Definition -------------------------
    # load the EfficientNet-B0, ensuring the head FC layer sets are left ff

    headmodel = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=Input(shape=(224,224,3)),
        classes=1000,
        classifier_activation="relu",
    )
    # construct the head of the model that will be placed on top of the
    # the base model
    model = headmodel.output
    model = AveragePooling2D(pool_size=(5, 5))(model)
    model = Flatten(name="flatten")(model)
    model = Dense(512, activation="relu")(model)
    model = Dropout(0.4)(model)
    model = Dense(len(lb.classes_), activation="softmax")(model)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    moodel = Model(inputs=headmodel.input, outputs=model)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    for layer in headmodel.layers:
        layer.trainable = False

    print('Model Definition Done')

    #   ------------------------- Training Model via Transfer Learning -------------------------
    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable)
    opt = Adam(learning_rate=1e-3)
    moodel.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
    # train the head of the network for a few epochs (all other layers
    # are frozen) -- this will allow the new FC layers to start to become
    # initialized with actual "learned" values versus pure random
    H = moodel.fit(
        trainAug.flow(trainX, trainY, batch_size=64),
        steps_per_epoch=len(trainX) // 64,
        validation_data=valAug.flow(testX, testY),
        validation_steps=len(testX) // 64,
        epochs=20)

    print('Model Trained Done')

    #   ------------------------- Extending the Epoches -------------------------

    H1 = moodel.fit(
        trainAug.flow(trainX, trainY, batch_size=64),
        steps_per_epoch=len(trainX) // 64,
        validation_data=valAug.flow(testX, testY),
        validation_steps=len(testX) // 64,
        epochs=10)

    print('Epoches Extended')

    tf.keras.models.save_model(moodel,'/home/vatsal/Desktop/AI-CC/AI_Auto_Training/models/Wheat_prediction.h5',overwrite=True)
