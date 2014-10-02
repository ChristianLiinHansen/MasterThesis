# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:24:49 2014

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

#For the trackbar
def nothing(x):
    pass

def GetCountours(image):
    
    #Use the FindCountours from OpenCV libraries
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
    
    #Do the raw moments to find the x,y coordinates
    centers = []
    radii = []
    print("Next image ...\n")
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1600 and area > 2200:
            #print("Area is too small")
            continue
        #else:
           # print("Area is:")
           # print(area)
        
        print("Area is: ",area)
        
        br = cv2.boundingRect(contour)
        radii.append(br[2])
        
        #Calculate the moments 
        m = cv2.moments(contour)
        if (int(m['m01']) == 0 or int(m['m00'] == 0)):
            continue
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
        
    return centers;
    


##########################################
# Main
##########################################

#fileName = '1.mp4'
fileName = 'DSC_0014.MOV'
video = cv2.VideoCapture(fileName)
#cap = cv2.VideoCapture('DSC_0005.MOV')     # 
#cap = cv2.VideoCapture('DSC_0006.MOV')     # 
#cap = cv2.VideoCapture('DSC_0007.MOV')     # 
#cap = cv2.VideoCapture('DSC_0008.MOV')     # 
#cap = cv2.VideoCapture('DSC_0009.MOV')     # 
#cap = cv2.VideoCapture('DSC_00010.MOV')    # Blue bricks only
#cap = cv2.VideoCapture('DSC_0013.MOV')      # Green and blue bricks
#video = cv2.VideoCapture(0)      # Test video loop

maxValue_trackbar = 255
threshold_trackbar = 20

# define range of blue color in HSV
lower_blue = np.array([110, 50, 50], dtype=np.uint8)
upper_blue = np.array([130,255,255], dtype=np.uint8)

# define range of green color in HSV
lower_green = np.array([60, 50, 50], dtype=np.uint8)
upper_green = np.array([95,255,255], dtype=np.uint8)

# define range of red color in HSV
lower_red = np.array([0, 50, 50], dtype=np.uint8)
upper_red = np.array([9,255,255], dtype=np.uint8)

# define range of yellow color in HSV
lower_yellow = np.array([16, 50, 50], dtype=np.uint8)
upper_yellow = np.array([27,255,255], dtype=np.uint8)

while(1):
    
    #Read each image and store it in "frame"
    sucessfully_read, frame = video.read()
   # width = cap.get(cv.CV_CAP_PROP_FPS)    
    #print("width is:", width)
    if not sucessfully_read:
        print("Video ended. Reloading video...")
        #cap = cv2.VideoCapture(video)   # works but is slow
        #frame = cap.read()
        #cap.set(cv.CV_CAP_PROP_POS_MSEC, 0)
        #cap.set(cv.CV_CAP_PROP_POS_FRAMES, 0)
        video.set(cv.CV_CAP_PROP_POS_AVI_RATIO, 0)        
        continue;
    
    # Resize each frames
    if video != 0 or video != 1:
        factor = 0.5
        smallImg = cv2.resize(frame, (0,0), fx=factor, fy=factor)
    else:
        #so it was either 1 or 0, means webcam or usb cam --> no downscaling
        smallImg = frame 
    
    # The the HSV image out from the RGB image
    hsv_img = cv2.cvtColor(smallImg, cv2.COLOR_BGR2HSV)
     
    # Create a black image, a window. Otherwise the trackbar would not work...
    cv2.namedWindow('red')    
    #cv2.namedWindow('green')      
    cv2.namedWindow('blue')  
    cv2.namedWindow('yellow')  

    # create trackbars for the red color
    cv2.createTrackbar('Hue min','Color segmented video - red',0,maxValue_trackbar,nothing)
    cv2.createTrackbar('Hue max','Color segmented video - red',9,maxValue_trackbar,nothing)  

    # create trackbars for the green color
    #cv2.createTrackbar('Hue min','Color segmented video - green',60,maxValue_trackbar,nothing)
    #cv2.createTrackbar('Hue max','Color segmented video - green',95,maxValue_trackbar,nothing)

    # create trackbars for the blue color
    cv2.createTrackbar('Hue min','Color segmented video - blue',100,maxValue_trackbar,nothing)
    cv2.createTrackbar('Hue max','Color segmented video - blue',120,maxValue_trackbar,nothing)
    
    # create trackbars for the blue color
    cv2.createTrackbar('Hue min','Color segmented video - yellow',22,maxValue_trackbar,nothing)
    cv2.createTrackbar('Hue max','Color segmented video - yellow',26,maxValue_trackbar,nothing)    
    
    # Threshold the HSV image to get only red colors
    red_img = cv2.inRange(hsv_img, lower_red, upper_red) 
    cv2.imshow('Color segmented video - red',red_img)  
    
    # Threshold the HSV image to get only blue colors
    blue_img = cv2.inRange(hsv_img, lower_blue, upper_blue) 
    cv2.imshow('Color segmented video - blue',blue_img)
    
    # Threshold the HSV image to get only green colors    
    green_img = cv2.inRange(hsv_img, lower_green, upper_green)
    #cv2.imshow('Color segmented video - green',green_img)

    # Threshold the HSV image to get only green colors    
    yellow_img = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    cv2.imshow('Color segmented video - yellow',yellow_img)        
   
    #Do a little morphology - Do some closing, i.e. erode and then dialate
    kernel = np.ones((3,3),np.uint8)
    mask_red = red_img    
    #mask_green = green_img
    mask_blue = blue_img
    mask_yellow = yellow_img
       
    #Do a little morphology - erosion
    iterations_erode = 2
    mask_red = cv2.erode(mask_red,kernel,iterations = iterations_erode)   
    #mask_green = cv2.erode(mask_green,kernel,iterations = iterations_erode)    
    mask_blue = cv2.erode(mask_blue,kernel,iterations = iterations_erode)
    mask_yellow = cv2.erode(mask_yellow,kernel,iterations = iterations_erode)
    
    #Do a little morphology - dialate
    iterations_dialate = 2
    mask_red = cv2.dilate(mask_red,kernel,iterations = iterations_dialate)  
    #mask_green = cv2.dilate(mask_green,kernel,iterations = iterations_dialate)      
    mask_blue = cv2.dilate(mask_blue,kernel,iterations = iterations_dialate)
    mask_yellow = cv2.dilate(mask_yellow,kernel,iterations = iterations_dialate)
      
    cv2.imshow('Opening video - red',mask_red)
    #cv2.imshow('Opening video - green',mask_green)    
    cv2.imshow('Opening video - blue',mask_blue)
    cv2.imshow('Opening video - yellow',mask_yellow)
    
    # get current positions of trackbar of green
    hueMin_trackbar_red = cv2.getTrackbarPos('Hue min','Color segmented video - red')
    hueMax_trackbar_red = cv2.getTrackbarPos('Hue max','Color segmented video - red')
    lower_red[0] = hueMin_trackbar_red
    upper_red[0] = hueMax_trackbar_red    
    
    """
    # get current positions of trackbar of green
    hueMin_trackbar_green = cv2.getTrackbarPos('Hue min','Color segmented video - green')
    hueMax_trackbar_green = cv2.getTrackbarPos('Hue max','Color segmented video - green')
    lower_green[0] = hueMin_trackbar_green
    upper_green[0] = hueMax_trackbar_green
    """
    
    # get current positions of trackbar of blue
    hueMin_trackbar_blue = cv2.getTrackbarPos('Hue min','Color segmented video - blue')
    hueMax_trackbar_blue = cv2.getTrackbarPos('Hue max','Color segmented video - blue')
    lower_blue[0] = hueMin_trackbar_blue
    upper_blue[0] = hueMax_trackbar_blue
    
    # get current positions of trackbar of yellow
    hueMin_trackbar_yellow = cv2.getTrackbarPos('Hue min','Color segmented video - yellow')
    hueMax_trackbar_yellow = cv2.getTrackbarPos('Hue max','Color segmented video - yellow')
    lower_yellow[0] = hueMin_trackbar_yellow
    upper_yellow[0] = hueMax_trackbar_yellow
    
    #Do a little morphology - Do some closing, i.e. erode and then dialate
    #kernel = np.ones((5,5),np.uint8)
    #erosion = cv2.erode(blue_img,kernel,iterations = 2)
    #cv2.imshow('Erosion video',erosion)
    #Do a little morphology - then dialate
    #dilation = cv2.dilate(erosion,kernel,iterations = 2)
    #cv2.imshow('Dilation video',dilation)
    
    # Finding the countours for red    
    centers_red = GetCountours(red_img)
    
    # Finding the countours for green    
    #centers_green = GetCountours(green_img)
    
    # Finding the countours for blue 
    centers_blue = GetCountours(blue_img)

    # Finding the countours for blue 
    centers_yellow = GetCountours(yellow_img)


    # Draw the countours on the input images, -1 is draw all countours
    #cv2.drawContours(smallImg, contours_blue, -1, (0,255,0), 3)

    #print("There are {} objects for blue".format(len(centers_blue)))
    #print("There are {} objects for green".format(len(centers_green)))

    # Color the central coordinates for red bricks with a filled circle
    for center in centers_red:
        cv2.circle(smallImg, center, 5, (0, 0, 255), -1)

    # Color the central coordinates for green bricks with a filled circle
    #for center in centers_green:
    #   cv2.circle(smallImg, center, 5, (0, 255, 0), -1)

    # Color the central coordinates for blue bricks with a filled circle
    for center in centers_blue:
        cv2.circle(smallImg, center, 5, (255, 255, 100), -1)
        
    # Color the central coordinates for yellow bricks with a filled circle
    for center in centers_yellow:
        cv2.circle(smallImg, center, 5, (0, 255, 255), -1)
    
    #cv2.waitKey(500)
   # print("Wait for 0.5 sec")
    # Execute if user hit Esc key
    cv2.imshow('Input video',smallImg)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("User closed the program")
        break 

video.release()
cv2.destroyAllWindows()