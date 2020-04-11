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
import cv2
import os
        
class GTSRBData:
    def __init__(self, basePath, whitelist):
        (self.TrainImages, self.TrainClasses) = loadDataset(basePath, "Train", whitelist)
        (self.TestImages, self.TestClasses) = loadDataset(basePath, "Test", whitelist)
        self.classTotals = self.TrainClasses.sum(axis=0)
        self.classWeight = self.classTotals.max() / self.classTotals


#class containing data and image referenced by one line of the GTSDB csv
class GTSRBImage:
    #preprocessing performed on every training and test image
    def processImage(self):
        pic = io.imread(self.imageFileName)
        #use the annotation file to crop the image
        pic = pic[self.roiX1:self.roiX2, self.roiY1:self.roiY2]
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
        return pic
    
    def __init__(self,args, basePath, whitelist):
        #parse line in annotation file to get the coordinates for cropping
        #the filename and the classID
        self.width = int(args[0])
        self.height = int(args[1])
        self.roiX1 = int(args[2])
        self.roiY1 = int(args[3])
        self.roiX2 = int(args[4])
        self.roiY2 = int(args[5])
        self.classID = whitelist.index(int(args[6]))
        self.imageFileName = os.path.join(basePath, args[7])
        self.image = self.processImage()
                


def loadDataset(basePath, dataset, whitelist):
    annotationFile = os.path.join(basePath, dataset + '.csv' )
    #get list of rows in annotation file
    rows = open(annotationFile).read().strip().split("\n")[1:]
    #randomize rows 
    np.random.shuffle(rows)
    
    GTSRBdata = []
    
    #iterate over rows in annotation file
    for i, row in enumerate(rows):
        if i%1000 == 0:
            print("processed " + str(i))
        #split up line into the individual rows
        args = row.strip().split(",")
        #only grab images that are present in the whitelist
        if int(args[6]) in whitelist:    
            GTSRBdata.append(GTSRBImage(args, basePath, whitelist))

    images = np.array([datum.image for datum in GTSRBdata])
    classes = np.array([datum.classID for datum in GTSRBdata])
    
    # one-hot encode the training and testing labels
    classes = to_categorical(classes)
    
    return (images, classes)
