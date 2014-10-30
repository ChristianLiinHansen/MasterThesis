# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:32:29 2014

@author: christian
"""

import numpy as np
import cv2
import cv
import math
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
    
        # Get the moments for each contour 
        m = cv2.moments(contour)
    
        # Get the Hu moments  for each contour 
        hu_m = cv2.HuMoments(m)
        
        #Store each parameters in a vector and return that vector
        parameters.append(compactness)  # Store compactness as 1. element.
        parameters.append(area)         # Store compactness as 2. element.
        parameters.append(perimeter)    # Store perimeter as 3. element.    
        parameters.append(hu_m[0])      # Store hu_one as 4. element.
        parameters.append(hu_m[1])      # Store hu_two as 5. element.
        parameters.append(hu_m[2])      # Store hu_three as 6. element.
        parameters.append(hu_m[3])      # Store hu_four as 7. element.
        parameters.append(hu_m[4])      # Store hu_five as 8. element.
        parameters.append(hu_m[5])      # Store hu_six as 9. element.
        parameters.append(hu_m[6])      # Store hu_seven as 10. element.
    return parameters

#def GetHuMoments(contour):
#    m = cv2.moments(contour)
#    print("The moment vector is: ", m)
#    print("\n")
#    
#    # To make it sense, 
#    hu_m = cv2.HuMoments(m)
#    print("The hu moment vector is with log: ", np.log(hu_m))
#    print("The hu moment vector is without log: ", hu_m)
#    print("\n")
#    return parameters    
##########################################q
# Main
##########################################
video = 'DSC_0015.MOV'
cap = cv2.VideoCapture(video)     

#Here the training data will be loaded....
compactness = []
area = []
perimeter = []
hu_one = []
hu_two = []
hu_tree = []
hu_four = []
hu_five = []
hu_six = []
hu_seven = []
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
    parameters = AnalyseContours(contours)
    
    compactness.append(parameters[0])    
    area.append(parameters[1])
    perimeter.append(parameters[2])
    hu_one.append(parameters[3])
    hu_two.append(parameters[4])   
    hu_tree.append(parameters[5])
    hu_four.append(parameters[6]) 
    hu_five.append(parameters[7]) 
    hu_six.append(parameters[8])
    hu_seven.append(parameters[9]) 

    #print("Compactness= ", parameters[0], "Area =", parameters[1])
    #print(parameters[0], parameters[1])
    # Wait until user hits the some key
    #cv2.waitKey(0)

#Taking from http://matplotlib.org/users/pyplot_tutorial.html
plt.plot(area, compactness, 'ro')
plt.title('Feature space')
plt.xlabel('area')
plt.ylabel('compactness')
plt.show()

plt.plot(hu_one, hu_tree, 'bo')
plt.title('Feature space')
plt.xlabel('hu_one')
plt.ylabel('hu_tree')
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