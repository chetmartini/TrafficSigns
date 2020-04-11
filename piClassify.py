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
    #resize the image to 32x32, since thats the size of the input
    #to the CNN
    pic = transform.resize(pic,(32,32))
    #scale the image to the range [0 1]
    pic = pic / 255.0
    pic = np.clip(pic,0,1)
    #contrast limited adaptive histogram equalization (CLAHE)
    #https://scikit-image.org/docs/dev/api/skimage.exposure.html#equalize-adapthist
    pic = exposure.equalize_adapthist(pic)
    #the resulting picture should be multiplied by 255 if it is to be 
    #saved to a png
    picInput = np.asarray([pic])
    return model.predict(picInput)

def getAnnotationText(pic):
    whitelist = [0,9,10,34,37,32,15]
    labelNames = getLabels("GTSRBsignnames.csv", whitelist)
    model = load_model("model.h5")
    prediction = makePrediction(model,pic)
    #prediction will be an array of floats the size of the output of the CNN
    #the max value will determine its prediction
    maxindex = np.argmax(prediction)
    maxval = prediction[0][maxindex]
    label = list(labelNames.keys())[list(labelNames.values()).index(maxindex)]
    return "%s (%3.2f)"%(label, maxval)

if __name__ == "__main__":
    #add png file path here
    pic = loadPic("ImageDataGeneratorExamples/15.png")
    label = getAnnotationText(pic)
    print(label)
