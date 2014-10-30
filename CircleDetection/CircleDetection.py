# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:10:08 2014

@author: christian
"""

import numpy as np
import cv2
import cv

##########################################
# Functions
##########################################
def GetCountours(image):
    
    #Use the FindCountours from OpenCV libraries
    contours, hierarchy = cv2.  findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)       
    return contours 

def AnalyseContours(contours, minArea, maxArea, minCompactness):
    
    #Do the raw moments to find the x,y coordinates
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed.
        
        #Be sure that if the calculating of area is 0, then skip this, so we dont devide by zero 
        # in the compactness calculation.
        if (area <= minArea) or (area > maxArea):
            continue

        compactness = (4 * 3.141592 * area) / (perimeter * perimeter) # If this is 1, a perfect circleis there.
        
        if compactness < minCompactness:
            continue
        
        print("compactness is:", compactness)
    
        #else:
            # print("Area is:")
            # print(area)
    
        #Calculate the moments 
        m = cv2.moments(contour)
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)    
    return centers;
 
##########################################q
# Main
##########################################

video = 'DSC_0015.MOV'
#video = 0                                    # webcam
#cap = cv2.VideoCapture('DSC_0005.MOV')     # 
#cap = cv2.VideoCapture('DSC_0006.MOV')     # 
#cap = cv2.VideoCapture('DSC_0007.MOV')     # 
#cap = cv2.VideoCapture('DSC_0008.MOV')     # 
#cap = cv2.VideoCapture('DSC_0009.MOV')     # 
#cap = cv2.VideoCapture('DSC_00010.MOV')    # Blue bricks only
#cap = cv2.VideoCapture('DSC_0013.MOV')      # Green and blue bricks

cap = cv2.VideoCapture(video)      # Test video loop
#Minimum area for being considered as circle
min_circle_area=200

#Margin of compactness for being considered as circle
margin=0.5

while(1):
    
    print("Next image ...\n")
    #Read each image and store it in "frame"
    sucessfully_read, frame = cap.read()
   # width = cap.get(cv.CV_CAP_PROP_FPS)    
    #print("width is:", width)
    if not sucessfully_read:
        print("Video ended. Reloading video...")
        #cap = cv2.VideoCapture(video)   # works but is slow
        #frame = cap.read()
        #cap.set(cv.CV_CAP_PROP_POS_MSEC, 0)
        #cap.set(cv.CV_CAP_PROP_POS_FRAMES, 0)
        cap.set(cv.CV_CAP_PROP_POS_AVI_RATIO, 0)        
        continue;
    
    # Resize each frames
    if video != 0 or video != 1:
        factor = 0.5
        smallImg = cv2.resize(frame, (0,0), fx=factor, fy=factor)
    else:
        #so it was either 1 or 0, means webcam or usb cam --> no downscaling
        smallImg = frame 
    
    # Get the grayscale image
    grayImg = cv2.cvtColor(smallImg, cv2.COLOR_BGR2GRAY) 
    
    # Do the thresholding with trackbar
    ret, threshold_img = cv2.threshold(grayImg,50,255,cv2.THRESH_BINARY_INV)

    #Do a little morphology - Do some closing, i.e. erode and then dialate
    kernel = np.ones((5,5),np.uint8)
    mask = threshold_img
    
    #Do a little morphology - dialate
    iterations_dialate = 1
    mask = cv2.dilate(mask,kernel,iterations = iterations_dialate)
    
    #Do a little morphology - erosion
    iterations_erode = 1
    mask = cv2.erode(mask,kernel,iterations = iterations_erode)    
    
    cv2.imshow('Thresholded video',mask)
    
    # Draw the countours on the input images, -1 is draw all countours
    cv2.drawContours(smallImg, contours, -1, (0,255,0), 3)
       
    
    contours = GetCountours(mask)
    
    #Analyse the contours for area
    minArea = 200
    maxArea = 2000
    minCompactness = 0.5
    centers = AnalyseContours(contours, minArea, maxArea, minCompactness)
    
    print("There are {} objects in the frame".format(len(centers)))
  
    
    # Color the central coordinates for blue bricks with a filled circle
    for center in centers:
        cv2.circle(smallImg, center, 5, (255, 255, 100), -1)
        

    # Execute if user hit Esc key
    cv2.imshow('Input video',smallImg)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()