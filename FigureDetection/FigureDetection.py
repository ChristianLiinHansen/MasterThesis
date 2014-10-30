# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:29:21 2014

@author: christian
"""

##########################################
# Libraries
##########################################
import numpy as np
import cv2
import matplotlib.pyplot as plt # required for plotting


#Load the data in
training1Img = cv2.imread("roundObjects.png", cv2.CV_LOAD_IMAGE_COLOR)
training2Img = cv2.imread("squares.png", cv2.CV_LOAD_IMAGE_COLOR)
training3Img = cv2.imread("triangles.png", cv2.CV_LOAD_IMAGE_COLOR)

# Do the grayscale converting   
grayImg1 = cv2.cvtColor(training1Img, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(training2Img, cv2.COLOR_BGR2GRAY) 
grayImg3 = cv2.cvtColor(training3Img, cv2.COLOR_BGR2GRAY)

#Do the thresholding
ret,thresh1 = cv2.threshold(grayImg1,220,255,cv2.THRESH_BINARY_INV)
ret,thresh2 = cv2.threshold(grayImg2,220,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(grayImg3,220,255,cv2.THRESH_BINARY_INV)

# Contour filling, maybee?

# Finding the contours
# Before applying the findContours, we need to clone the image, otherwise
# we mess with the original image
contourImage1 = thresh1.copy()
contourImage2 = thresh2.copy()
contourImage3 = thresh3.copy()
contours1, hierarchy = cv2.findContours(contourImage1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
contours2, hierarchy = cv2.findContours(contourImage2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
contours3, hierarchy = cv2.findContours(contourImage3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

#Draw the contours
cv2.drawContours(training1Img,contours1,-1,(0,255,0),2)
cv2.drawContours(training2Img,contours2,-1,(0,255,0),2)
cv2.drawContours(training3Img,contours3,-1,(0,255,0),2)

#Create the list for each image
area1 = []
perimeter1 = []
compactness1 = []

area2 = []
perimeter2 = []
compactness2 = []

area3 = []
perimeter3 = []
compactness3 = []
   
for contour in contours1:
    #Get the area, perimeter and compactness of the contours in contours1    
    temp_area = cv2.contourArea(contour,False)
    
    #Skip the iteration if the area is less than something     
    if temp_area < 10:
        continue 
    temp_perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed. 
    temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)
    
    area1.append(temp_area)   
    perimeter1.append(temp_perimeter)
    compactness1.append(temp_compactness)

for contour in contours2:
    #Get the area, perimeter and compactness of the contours in contours1    
    temp_area = cv2.contourArea(contour,False) 
    
    #Skip the iteration if the area is less than something 
    if temp_area < 10:
        continue     
    
    temp_perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed. 
    temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)
    
    area2.append(temp_area)   
    perimeter2.append(temp_perimeter)
    compactness2.append(temp_compactness)

for contour in contours3:
    #Get the area, perimeter and compactness of the contours in contours1    
    temp_area = cv2.contourArea(contour,False) 
    
    #Skip the iteration if the area is less than something 
    if temp_area < 10:
        continue 
    
    temp_perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed. 
    temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)
    
    area3.append(temp_area)   
    perimeter3.append(temp_perimeter)
    compactness3.append(temp_compactness)
    
#So now we have tree features from tree images
# Let plot the featuers in a 2D plot.
plt.figure(1)
plt.plot(area1, perimeter1, 'ro')
plt.plot(area2, perimeter2, 'go')
plt.plot(area3, perimeter3, 'bo')

plt.title('Feature space')
plt.xlabel('Area')
plt.ylabel('Perimeter')

plt.figure(2)
plt.plot(area1, compactness1, 'ro')
plt.plot(area2, compactness2, 'go')
plt.plot(area3, compactness3, 'bo')

#Plot the mean of the cluster
plt.plot(np.mean(area1), np.mean(compactness1), 'rs', markersize=20)
#Plot the mean of the cluster
plt.plot(np.mean(area2), np.mean(compactness2), 'gs', markersize=20)
#Plot the mean of the cluster
plt.plot(np.mean(area3), np.mean(compactness3), 'bs', markersize=20)

plt.title('Feature space')
plt.xlabel('Area')
plt.ylabel('Compactness')

plt.figure(3)
plt.plot(perimeter1, compactness1, 'ro')
plt.plot(perimeter2, compactness2, 'go')
plt.plot(perimeter3, compactness3, 'bo')
plt.title('Feature space')
plt.xlabel('Perimeter')
plt.ylabel('Compactness')

#And then show all the figures
#plt.show()


# Now the classifier can start. First I will try with the K-means algorithm


#f, subfig = plt.subplots(2, 3)
#subfig[0, 0].plot(area1, compactness1, 'ro')
#subfig[0, 0].set_title('Axis [0,0]')
#subfig[0, 1].plot(area1, compactness1, 'go')
#subfig[0, 1].set_title('Axis [0,1]')
#subfig[1, 0].plot(area1, compactness1, 'ro')
#subfig[1, 0].set_title('Axis [1,0]')
#subfig[1, 1].plot(area1, compactness1, 'ro')
#subfig[1, 1].set_title('Axis [1,1]')

#plt.plot(area1, compactness1, 'ro')
#plt.plot(area2, compactness2, 'go')
#plt.plot(area3, compactness3, 'bo')
#plt.title('Feature space')

while(1):
    cv2.imshow("training1Img", training1Img)
    cv2.imshow("trainin2Img", training2Img)
    cv2.imshow("trainin3Img", training3Img)    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("User closed the program...")
        break  

cv2.destroyAllWindows()