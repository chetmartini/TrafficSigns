# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:04:51 2020

@author: AANDE355
"""
from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
from tensorflow.keras.utils import to_categorical
import os
        
class GTSRBData:
    def __init__(self, basePath):
        (self.TrainImages, self.TrainClasses) = loadDataset(basePath, "Train")
        (self.TestImages, self.TestClasses) = loadDataset(basePath, "Test")
        self.classTotals = self.TrainClasses.sum(axis=0)
        self.classWeight = self.classTotals.max() / self.classTotals


#class containing data and image referenced by one line of the GTSDB csv
class GTSRBImage:
    def processImage(self):
        pic = io.imread(self.imageFileName)
        pic = pic[self.roiX1:self.roiX2, self.roiY1:self.roiY2]
        pic = transform.resize(pic,(32,32))
        pic = exposure.equalize_adapthist(pic)
        return pic
    
    def __init__(self,CSVLine, basePath):
        args = CSVLine.strip().split(",")
        self.width = int(args[0])
        self.height = int(args[1])
        self.roiX1 = int(args[2])
        self.roiY1 = int(args[3])
        self.roiX2 = int(args[4])
        self.roiY2 = int(args[5])
        self.classID = int(args[6])
        self.imageFileName = os.path.join(basePath, args[7])
        self.image = self.processImage()
                


def loadDataset(basePath, dataset):
    
    annotationFile = os.path.join(basePath, dataset + '.csv' )
    rows = open(annotationFile).read().strip().split("\n")[1:]
    np.random.shuffle(rows)
    GTSRBdata = []
    
    for i, row in enumerate(rows):
        if i>5000:
            break
        if i%1000 == 0:
            print("processed " + str(i))
        GTSRBdata.append(GTSRBImage(row, basePath))

    images = np.array([datum.image for datum in GTSRBdata])
    # scale data to the range of [0, 1]
    images = images.astype("float32") / 255.0
    classes = np.array([datum.classID for datum in GTSRBdata])
    
    # one-hot encode the training and testing labels
    #numClasses = len(np.unique(classes))
    classes = to_categorical(classes)
    return (images, classes)
