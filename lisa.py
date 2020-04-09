from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

class LISAData:
    def __init__(self, basePath, classIDs):
        (self.TrainImages,
         self.TrainClasses,
         self.TestImages,
         self.TestClasses) = loadData(basePath, classIDs)
        self.classTotals = self.TrainClasses.sum(axis=0)
        self.classWeight = self.classTotals.max() / self.classTotals
        
#class containing data and image referenced by one line of the GTSDB csv
class LISAImage:
    
    def processImage(self):
        pic = io.imread(self.imageFileName)
        pic = pic[self.roiY1:self.roiY2, self.roiX1:self.roiX2 ]
        pic = transform.resize(pic,(32,32))
        pic = exposure.equalize_adapthist(pic)
        return pic
    
    def __init__(self,CSVLine, basePath, classIDs):
        args = CSVLine.strip().split(";")
        self.imageFileName = os.path.join(basePath, args[0])
        self.classID = classIDs[args[1]]
        self.roiX1 =int(args[2])
        self.roiY1 = int(args[3])
        self.roiX2 = int(args[4])
        self.roiY2 = int(args[5])
        #self.occluded = int(args[6])
        self.image = self.processImage()
                

def loadData(basePath, classIDs):
    annotationFile = os.path.join(basePath, 'allAnnotations.csv' )
    rows = open(annotationFile).read().strip().split("\n")[1:]
    np.random.shuffle(rows)
    LISAdata = []
    
    for i, row in enumerate(rows):
        if i%1000 == 0:
            print("processed " + str(i))
        LISAdata.append(LISAImage(row, basePath, classIDs))

    images = np.array([datum.image for datum in LISAdata])
    # scale data to the range of [0, 1]
    images = images.astype("float32") / 255.0
    trainImages = images[0::2]
    testImages = images[1::2]
    classes = np.array([datum.classID for datum in LISAdata])    
    classes = to_categorical(classes)
    trainClasses = classes[0::2]
    testClasses = classes[1::2]
    
    return (trainImages,trainClasses,testImages,testClasses)