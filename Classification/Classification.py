# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:32:29 2014

@author: christian
"""

import numpy as np
import cv2
import cv
from matplotlib import pyplot as plt

##########################################
# Functions
##########################################
def ResizeImg(img,scale):
    img_resize = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return img_resize

def GetCountours(image):
    
    #Use the FindCountours from OpenCV libraries
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    return contours 

def AnalyseContours(contours):
    parameters = []
    for contour in contours:
        # Get the area of the contour
        area = cv2.contourArea(contour)
        
        #Get the perimeter of the contour
        perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed.  
    
        # Get the compactness of the contour    
        compactness = (4 * 3.141592 * area) / (perimeter * perimeter) # If this is 1, a perfect circleis there.
    
        #Store each parameters in a vector and return that vector
        parameters.append(compactness)
        parameters.append(area)        
    return parameters;
 
##########################################q
# Main
##########################################
video = 'DSC_0015.MOV'
cap = cv2.VideoCapture(video)     

#Here the training data will be loaded....
area = []
compactness = []
for i in range(1,34):
    # "1.png" to "30.png" is training data, where "31,32,33.png" is the star, square and triangle
    stringName = str(i) + ".png"
    #print(stringName)
    img =  cv2.imread(stringName, cv.CV_LOAD_IMAGE_COLOR)
    #cv2.imshow(stringName, img)     
    
    # Do the grayscale converting    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Do the thresholding    
    ret, threshold_img = cv2.threshold(grayImg,50,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow("Thresholded image", threshold_img)

    #Get the contours in the image
    contours = GetCountours(threshold_img)

    #Analysing the contours
    parameters =AnalyseContours(contours)
    
    area.append(parameters[0])
    compactness.append(parameters[1])
    
    #print("Compactness= ", parameters[0], "Area =", parameters[1])
    print(parameters[0], parameters[1])
# Wait until user hits the some key
cv2.waitKey(0)

#Taking from http://matplotlib.org/users/pyplot_tutorial.html
plt.plot(area, compactness, 'ro')
plt.title('Feature space')
plt.xlabel('Compactness')
plt.ylabel('Area')
plt.show()

#Now we play the video and see if the system has learn something from the
#training data.

print("Playing the video...")
while(1):    
    sucessfully_read, frame = cap.read()
    if not sucessfully_read:
        print("Video ended. Reloading video...")
        cap.set(cv.CV_CAP_PROP_POS_AVI_RATIO, 0)        
        continue
    
    #Resize the image by a given scale, 1 is 100%, 0.4 is 40%, etc. 
    scale = 0.5
    img_resize = ResizeImg(frame,scale)    
    
    cv2.imshow("Result of learning system", img_resize)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("User closed the program...")
        break  
cap.release()
cv2.destroyAllWindows()