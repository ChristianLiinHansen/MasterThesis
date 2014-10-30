# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 19:54:00 2014

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
import os

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
        
    elif(color == "yellow"):
        # get current "hue" position of trackbar of yellow
        lower_yellow[0] = cv2.getTrackbarPos('Hue min','Color segmented - yellow')
        upper_yellow[0] = cv2.getTrackbarPos('Hue max','Color segmented - yellow')
        
        # get current "saturation" position of trackbar of yellow
        lower_yellow[1] = cv2.getTrackbarPos('Saturation min','Color segmented - yellow')
        upper_yellow[1] = cv2.getTrackbarPos('Saturation max','Color segmented - yellow')
        
        # get current "value" position of trackbar of yellow
        lower_yellow[2] = cv2.getTrackbarPos('Value min','Color segmented - yellow')
        upper_yellow[2] = cv2.getTrackbarPos('Value max','Color segmented - yellow')
        
    elif(color == "blue"):
        # get current "hue" position of trackbar of blue
        lower_blue[0] = cv2.getTrackbarPos('Hue min','Color segmented - blue')
        upper_blue[0] = cv2.getTrackbarPos('Hue max','Color segmented - blue')
        
        # get current "saturation" position of trackbar of blue
        lower_blue[1] = cv2.getTrackbarPos('Saturation min','Color segmented - blue')
        upper_blue[1] = cv2.getTrackbarPos('Saturation max','Color segmented - blue')
        
        # get current "value" position of trackbar of blue
        lower_blue[2] = cv2.getTrackbarPos('Value min','Color segmented - blue')
        upper_blue[2] = cv2.getTrackbarPos('Value max','Color segmented - blue')
    else:
        print("Choose between red, yellow or blue")

def CreateTrackBarRed():
    cv2.namedWindow('Color segmented - red')
    cv2.createTrackbar('Hue min','Color segmented - red',0,255,nothing)
    cv2.createTrackbar('Hue max','Color segmented - red',20,255,nothing)
    cv2.createTrackbar('Saturation min','Color segmented - red',157,255,nothing)
    cv2.createTrackbar('Saturation max','Color segmented - red',255,255,nothing)   
    cv2.createTrackbar('Value min','Color segmented - red',136,255,nothing)
    cv2.createTrackbar('Value max','Color segmented - red',255,255,nothing)

def CreateTrackBarYellow():
    cv2.namedWindow('Color segmented - yellow')
    cv2.createTrackbar('Hue min','Color segmented - yellow',20,255,nothing)
    cv2.createTrackbar('Hue max','Color segmented - yellow',41,255,nothing)
    cv2.createTrackbar('Saturation min','Color segmented - yellow',166,255,nothing)
    cv2.createTrackbar('Saturation max','Color segmented - yellow',255,255,nothing)   
    cv2.createTrackbar('Value min','Color segmented - yellow',187,255,nothing)
    cv2.createTrackbar('Value max','Color segmented - yellow',255,255,nothing)
    
def CreateTrackBarBlue():
    cv2.namedWindow('Color segmented - blue')
    cv2.createTrackbar('Hue min','Color segmented - blue',60,255,nothing)
    cv2.createTrackbar('Hue max','Color segmented - blue',255,255,nothing)
    cv2.createTrackbar('Saturation min','Color segmented - blue',0,255,nothing)
    cv2.createTrackbar('Saturation max','Color segmented - blue',255,255,nothing)   
    cv2.createTrackbar('Value min','Color segmented - blue',0,255,nothing)
    cv2.createTrackbar('Value max','Color segmented - blue',255,255,nothing)
    
def GetCentersBlue(image):
    
    #Use the FindCountours from OpenCV libraries
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

    #Do the raw moments to find the x,y coordinates
    centers = []
    
    #Analyzing the size. Filter out the small noise pixels.
    for contour in contours:
        area = cv2.contourArea(contour)
        print("Blue area: ", area)
        if (area < 400) or (area > 1500):        
            continue
        #else:
            #fileObject.write(str(area)+"\n");
            #print("Area is:", area)

        #Calculate the moments 
        m = cv2.moments(contour)
        if (int(m['m01']) == 0 or int(m['m00'] == 0)):
            continue
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
    #fileObject.close()
    return centers

def GetCentersRed(image):
    
    #Use the FindCountours from OpenCV libraries
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

    #Do the raw moments to find the x,y coordinates
    centers = []
    
    #Analyzing the size. Filter out the small noise pixels.
    for contour in contours:
        area = cv2.contourArea(contour)
        print("Red area: ", area)
        if (area < 1000) or (area > 3000):        
            continue
        #else:
            #fileObject.write(str(area)+"\n");
            #print("Area is:", area)

        #Calculate the moments 
        m = cv2.moments(contour)
        if (int(m['m01']) == 0 or int(m['m00'] == 0)):
            continue
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
    #fileObject.close()
    return centers
    
def GetCentersYellow(image):
    
    #Use the FindCountours from OpenCV libraries
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

    #Do the raw moments to find the x,y coordinates
    centers = []
    
    #Analyzing the size. Filter out the small noise pixels.
    for contour in contours:
        area = cv2.contourArea(contour)
        print("Yellow area: ", area)        
        if (area < 1000) or (area > 4000):        
            continue
        #else:
            #fileObject.write(str(area)+"\n");
           # print("Area is:", area)

        #Calculate the moments 
        m = cv2.moments(contour)
        if (int(m['m01']) == 0 or int(m['m00'] == 0)):
            continue
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
    #fileObject.close()
    return centers; 
##########################################
# Main
##########################################

#fileName = 'test_with_light_sidebyside.mp4'
#fileName = 'test_without_light_in_rows.mp4'
#fileName = 'test_without_light_sidebyside.mp4'

#fileName = 'warm_ligt_slow.mp4'
fileName = -1  # -1 is the latest device that has been hooked up.
video = cv2.VideoCapture(fileName)

# define range of red color in HSV
lower_red = np.array([0, 157, 136], dtype=np.uint8)
upper_red = np.array([20,255,255], dtype=np.uint8)

# define range of yellow color in HSV
lower_yellow = np.array([20, 166, 187], dtype=np.uint8)
upper_yellow = np.array([41,255,255], dtype=np.uint8)

# define range of blue color in HSV
lower_blue = np.array([60, 0, 0], dtype=np.uint8)
upper_blue = np.array([255,255,255], dtype=np.uint8)

CreateTrackBarRed()
CreateTrackBarYellow()
CreateTrackBarBlue()

## Make sure that we use a clean text file each time we run this program
filename= "TextFile.txt"

## delete only if file exists ##
if os.path.exists(filename):
    print("The textfile: %s exist already and hence we remove it..." % filename)
    os.remove(filename)
else:
    print("The textfile: %s does not exist." % filename)

timeCounter = 0
while(1):
    
    #fileObject = open("TextFile.txt", "a") 
    #Read each image and store it in "frame"
    sucessfully_read, img = video.read()    

    
    #print("width is:", width)
    if not sucessfully_read:
        print("Video ended. Reloading video...")
        cv2.waitKey(1000)
        print("Is the webcam plugged in?...")
        timeCounter = timeCounter + 1
        if timeCounter > 5:
            print("Breaking the system...")
            break
        #cap = cv2.VideoCapture(video)   # works but is slow
        #frame = cap.read()
        #cap.set(cv.CV_CAP_PROP_POS_MSEC, 0)
        #cap.set(cv.CV_CAP_PROP_POS_FRAMES, 0)
        video.set(cv.CV_CAP_PROP_POS_AVI_RATIO, 0)        
        continue;

    #Read an image
    #img =  cv2.imread('lego.jpg', cv.CV_LOAD_IMAGE_COLOR) 
    TrackBarListener("red")
    TrackBarListener("yellow")
    TrackBarListener("blue")
    
    #Resize the image by a given scale, 1 is 100%, 0.4 is 40%, etc. 
    scale = 1
    img_resize = ResizeImg(img,scale)

    #cv2.imshow("Input image resized", img_resize)
    
    #Convert the image into a grayscale
    #img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grayscale image", img_gray)
    
    #Convert the image into hsv
    img_hsv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV image", img_hsv)
    
    #Convert to binary image using thresholding with colorsegmentation
    img_red = cv2.inRange(img_hsv, lower_red, upper_red)
    img_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  
    img_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)  
    
    # Subtract the blue and yellow to isolate the red
    #img_red = img_red - img_yellow - img_blue
    #cv2.imshow("Color segmented - red alone", img_red)
    
    # Write image
    #cv2.imwrite("img_red.png", img_red)
    #cv2.imwrite("img_yellow.png", img_yellow)
    #cv2.imwrite("img_blue.png", img_blue)      
    
    #Do a little morphology to remove the lastpart of the LEGO yellow LEGO brick
    iterations_erode = 3
    iterations_dialate = 2
    kernel = np.ones((3,3),np.uint8)
    
    img_red = cv2.erode(img_red,kernel,iterations = iterations_erode)    
    #cv2.imshow("Color segmented - red alone after erosion", img_red)
    img_red = cv2.dilate(img_red,kernel,iterations = iterations_dialate)
    #cv2.imshow("Color segmented - red alone after dialate", img_red)
    
    img_blue = cv2.erode(img_blue,kernel,iterations = iterations_erode)    
    #cv2.imshow("Color segmented - blue alone after erosion", img_blue)
    img_blue = cv2.dilate(img_blue,kernel,iterations = iterations_dialate)
    #cv2.imshow("Color segmented - blue alone after dialate", img_blue)
    
    img_yellow = cv2.erode(img_yellow,kernel,iterations = iterations_erode)    
    #cv2.imshow("Color segmented - yellow alone after erosion", img_yellow)
    img_yellow = cv2.dilate(img_yellow,kernel,iterations = iterations_dialate)
    #cv2.imshow("Color segmented - yellow alone after dialate", img_yellow)

    cv2.imshow("Color segmented - red", img_red)
    cv2.imshow("Color segmented - yellow", img_yellow)
    cv2.imshow("Color segmented - blue", img_blue)
    
    # Finding the center coordinates for red, yellow and blue    
    centers_red = GetCentersRed(img_red)
    centers_yellow = GetCentersYellow(img_yellow)
    centers_blue = GetCentersBlue(img_blue)

# Color the central coordinates for red bricks with a filled circle

    for center in centers_red:
       cv2.circle(img_resize, center, 5, (0, 0, 255), -1)
       cv2.circle(img_resize, center, 10, (0, 0, 0), 2)   
       print("x,y for red is:", center)

# Color the central coordinates for yellow bricks with a filled circle
    for center in centers_yellow:
        cv2.circle(img_resize, center, 5, (0, 255, 255), -1)
        cv2.circle(img_resize, center, 10, (0, 0, 0), 2)           
        print("x,y for yellow is:", center)
        
    # Color the central coordinates for blue bricks with a filled circle
    for center in centers_blue:
        cv2.circle(img_resize, center, 5, (255, 0, 0), -1)
        cv2.circle(img_resize, center, 10, (0, 0, 0), 2)        
        print("x,y for blue is:", center)
      
    # Remember to check if the vector is empty? Does the program crash or not?

      
    cv2.imshow("Tracking of red image resized", img_resize)
    print("Next image ...\n")    
    
    #Hit ECS to close the program
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("User closed the program...")
        break 

cv2.destroyAllWindows()

