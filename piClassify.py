# import the necessary packages
from tensorflow.keras.models import load_model
from skimage import exposure
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import cv2
import numpy as np

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

def loadPic(imageFileName):
    pic = io.imread(imageFileName)
    return pic

def makePrediction(model,pic):
    pic = np.ndarray.astype(pic, float)
    pic = transform.resize(pic,(32,32))
    pic = pic.astype("float32") / 255.0
    #pic = exposure.equalize_adapthist(pic)
    picInput = np.asarray([pic])
    return model.predict(picInput)

def getAnnotationText(pic):
    whitelist = [0,9,10,34,37,32,15]
    labelNames = getLabels("GTSRBsignnames.csv", whitelist)
    model = load_model("model.h5")
    prediction = makePrediction(model,pic)
    maxindex = np.argmax(prediction)
    maxval = prediction[0][maxindex]
    label = list(labelNames.keys())[list(labelNames.values()).index(maxindex)]
    return "%s (%3.2f)"%(label, maxval)

if __name__ == "__main__":
    pic = loadPic("x.png")
    cv2.imwrite("x2.png",pic)
    label = getAnnotationText(pic)
    print(label)
