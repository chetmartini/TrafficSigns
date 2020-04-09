# import the necessary packages
from tensorflow.keras.models import load_model
from skimage import exposure
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

def getLabels(csvFileName):
    labelLines = open(csvFileName).read().strip().split("\n")[1:]
    labelNames = {}
    for labelLine in labelLines:
        labelArgs = labelLine.split(",")
        labelName = labelArgs[1]
        labelID = labelArgs[0]
        labelNames[labelName] = labelID  
    return labelNames

def loadPic(imageFileName):
    pic = io.imread(imageFileName)
    pic = transform.resize(pic,(32,32))
    pic = exposure.equalize_adapthist(pic)
    #plt.figure(0)
    #plt.imshow(pic)
    return pic

def makePrediction(model,pic):
    picInput = np.asarray([pic])
    return model.predict(picInput)

labelNames = getLabels("GTSRBsignnames.csv")
model = load_model("model.h5")
pic = loadPic("gtsrbexample.png")
prediction = makePrediction(model,pic)
value = np.where(prediction == 1)[1][0]
print(list(labelNames.keys())[value])
