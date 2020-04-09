import cv2
import numpy as np
import matplotlib.pyplot as plt
from piClassify import getAnnotationText

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]

def getText(pic,circle):
    center = (circle[0],circle[1])
    radius = circle[2]+5
    topleft = (center[0]-radius,center[1]-radius)
    bottomright = (center[0]+radius, center[1]+radius)
    croppedPic = cropSign(pic,(topleft,bottomright))
    return getAnnotationText(croppedPic) #function from piClassify


def findSigns(pic):
    image = pic.copy()
    image = cv2.medianBlur(image,5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=0, maxRadius=0)
    return circles

def annotateSigns(pic, circles):
    if circles is None:
        return
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        text = getText(pic,i)
        cv2.putText(pic, text,(i[0]-i[2],i[1]-i[2]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv2.circle(pic, (i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(pic, (i[0],i[1]),2,(0,0,255),3)
        print("found sign at: %d,%d . type = %s "%(i[0],i[1],text) )
    return


pic = cv2.imread("detectexample.png")
pic = cv2.resize(pic, (640,480))
circles = findSigns(pic)
annotateSigns(pic, circles)
plt.imshow(pic)
cv2.imwrite("out.png",pic)