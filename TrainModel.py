# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from skimage import transform
from gtsrb import GTSRBData
from lisa import LISAData
import cv2

#parse csv file that associates sign names with their classID
#returns a dictionary
def getLabels(csvFileName, whitelist):
    labelLines = open(csvFileName).read().strip().split("\n")[1:]
    labelNames = {}
    for labelLine in labelLines:
        labelArgs = labelLine.split(",")
        labelName = labelArgs[1]
        labelID = int(labelArgs[0])
        if labelID in whitelist:    
            labelNames[labelName] = whitelist.index(labelID) 
    labelNames = {k: v for k, v in sorted(labelNames.items(), key=lambda item: item[1])}
    return labelNames
        
def LoadLISA():
    print("loading data from LISA database")
    labelNames = getLabels("LISAsignnames.csv")
    dataDirectory = os.path.abspath('signDatabasePublicFramesOnly')
    data = LISAData(dataDirectory, labelNames)
    return data, labelNames

def LoadGTSRB():
    #only use a subset of images for simplicity. these numbers are the 
    #classIDs of the signs as specified in the GTSRBsignnames.csv file
    whitelist = [0,9,10,34,37,32,15]
    print("loading data from GTSRB database")
    labelNames = getLabels("GTSRBsignnames.csv", whitelist)
    dataDirectory = os.path.abspath('gtsrb-german-traffic-sign')
    data = GTSRBData(dataDirectory, whitelist)
    return data, labelNames

#creates a sequential CNN model using the Keras library
def MakeModel(numOutputs):
    #dropout: https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
        # http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    #batch normalization: https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
        #need to decide between doing it before or after activation function
        # https://arxiv.org/pdf/1502.03167.pdf
    
    model = Sequential()
    #input is 32x32 RGB image
    inputShape = (32, 32, 3)
    
    #convolutional + pooling layer
    model.add(Conv2D(8, (5, 5), input_shape=inputShape, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #convolutional + pooling layer
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #fully-connected layer
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
         
    #fully-connected output layer
    model.add(Dense(numOutputs))
    model.add(Activation("softmax"))


    model.compile(loss="categorical_crossentropy", optimizer="Adam",
                  metrics=["accuracy"])
    return model

#get data from either LISA or GTSRB
data, labelNames = LoadGTSRB()
#data, labelNames = LoadLISA()

model = MakeModel(len(labelNames))

numEpochs = 20
batchSize = 64

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

# train the network
print("Training")
H = model.fit_generator(
	aug.flow(data.TrainImages, data.TrainClasses, batch_size=batchSize),
	validation_data=(data.TestImages, data.TestClasses),
	steps_per_epoch=data.TestImages.shape[0] // batchSize,
	epochs=numEpochs,
	class_weight=data.classWeight,
	verbose=1)

# evaluate the network
print("Evaluating")
predictions = model.predict(data.TestImages, batch_size=batchSize)
print(classification_report(data.TestClasses.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames.keys()))

print("saving model to model.h5")
model.save("model.h5")