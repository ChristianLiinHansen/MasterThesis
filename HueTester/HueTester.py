# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 21:20:04 2014

@author: christian
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 13:00:30 2014

@author: christian
"""

##########################################
# References
##########################################
# Trackbar
# http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html

# Tracking the circles
#http://stackoverflow.com/questions/21612258/filled-circle-detection-using-cv2-in-python

##########################################
# Libraries
##########################################
import numpy as np
import cv2
import cv

##########################################
# Functions
##########################################
def nothing(x):
    pass

def ResizeImg(img,scale):
    img_resize = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return img_resize

def TrackBarListener(color):

    if(color == "red"):
        # get current "hue" position of trackbar of red
        lower_red[0] = cv2.getTrackbarPos('Hue min','Color segmented - red')
        upper_red[0] = cv2.getTrackbarPos('Hue max','Color segmented - red')
        
        # get current "saturation" position of trackbar of red
        lower_red[1] = cv2.getTrackbarPos('Saturation min','Color segmented - red')
        upper_red[1] = cv2.getTrackbarPos('Saturation max','Color segmented - red')
        
        # get current "value" position of trackbar of red
        lower_red[2] = cv2.getTrackbarPos('Value min','Color segmented - red')
        upper_red[2] = cv2.getTrackbarPos('Value max','Color segmented - red')

def CreateTrackBarRed():
    cv2.namedWindow('Color segmented - red')
    cv2.createTrackbar('Hue min','Color segmented - red',0,255,nothing)
    cv2.createTrackbar('Hue max','Color segmented - red',255,255,nothing)
    cv2.createTrackbar('Saturation min','Color segmented - red',146,255,nothing)
    cv2.createTrackbar('Saturation max','Color segmented - red',255,255,nothing)   
    cv2.createTrackbar('Value min','Color segmented - red',0,255,nothing)
    cv2.createTrackbar('Value max','Color segmented - red',255,255,nothing)

##########################################
# Main
##########################################

# define range of red color in HSV
lower_red = np.array([0, 146, 0], dtype=np.uint8)
upper_red = np.array([255,255,255], dtype=np.uint8)

CreateTrackBarRed()

while(1):
    #Read an image
    img =  cv2.imread('kreisz.png', cv.CV_LOAD_IMAGE_COLOR) 
    TrackBarListener("red")
    
    #Resize the image by a given scale, 1 is 100%, 0.4 is 40%, etc. 
    scale = 1
    img_resize = ResizeImg(img,scale)
    cv2.imshow("Input image resized", img_resize)
    
    #Convert the image into a grayscale
    #img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grayscale image", img_gray)
    
    #Convert the image into hsv
    img_hsv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV image", img_hsv)
    
    #Convert to binary image using thresholding with colorsegmentation
    img_red = cv2.inRange(img_hsv, lower_red, upper_red)
    cv2.imshow("Color segmented - red", img_red)

    print("Next image ...\n")    
    
    #Hit ECS to close the program
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("User closed the program...")
        break 

cv2.destroyAllWindows()

