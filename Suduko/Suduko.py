# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:32:29 2014

@author: christian
"""

import numpy as np
import cv2
import cv

import matplotlib.patches as mpatches # For making legends http://matplotlib.org/users/legend_guide.html
import matplotlib.pyplot as plt

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
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
    #We look into how big the contours vector is....
    print("Size of contour is", len(contours))  
    
    #print contours  -- The contours is a list of list for each contours x,y pixel coordinate
    #To get the i,j'th element
    #print a.item(2,2) 
    
    #To set the i.j'th element
    #a[2,3] = 24
    #print a
    
    # The contours is a list of many different contour. 
    # So the first contour contains x,y pixel coordinates which bounds the first contour
    # And the contours does then contain all the different contour, i.e. 
    # contours = [contour1, contour2 ....]
    # where contour1 = [ [x1,y1] [x2,y2] ....]
   
    # So then we know how big our matrix should be....
    n = len(contours)
    numberOfFeatures = 4
    featuresMatrix = np.zeros(shape=(n,numberOfFeatures))   
    i = 0
    #print("Area \t Perimeter \t Compactness \t hu_1 \t hu_2 \t hu_3 \t hu_4 \t hu_5 \t hu_6 \t hu_7")
    
    for contour in contours:       
        # Get the area of the each contour
        area = cv2.contourArea(contour)
        #Get the perimeter of the contour
        perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed.  
         # Get the compactness of the contour    
        compactness = (4 * 3.141592 * area) / (perimeter * perimeter) # If this is 1, a perfect circleis there.
        
        # If the area is less than 200, we brake the for loop and continue with the next contour        
        # OR if the perimeter is less than 100, we brake the for loop and continue with the next contour                        
        if ((area < 200) or (perimeter < 100)):
            continue

        #Else we store the values into the featureMatrix
        featuresMatrix[i,0] = area
        featuresMatrix[i,1] = perimeter        
        
        #featuresMatrix[i,2] = compactness  
        
        # Get the moments for each contour. 
        #Note that m contains the raw/spatial moments m_ij
        #central moments mu_ij 
        #and normalized central moments nu_ij 
        # See http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments
        m = cv2.moments(contour)     

        # Get the Hu moments  for each contour 
        hu_m = cv2.HuMoments(m)
        
        featuresMatrix[i,2] = m['mu12']  
        
        #Store each hu_moment in the feature matrix        
        #featuresMatrix[i,3] = hu_m[0]        
        #featuresMatrix[i,4] = hu_m[1]
        #featuresMatrix[i,5] = hu_m[2]    
        #featuresMatrix[i,6] = hu_m[3]    
        #featuresMatrix[i,7] = hu_m[4]    
        #featuresMatrix[i,8] = hu_m[5]    
        #featuresMatrix[i,9] = hu_m[6]    
    
        #At the end of the loop we increment the counter i, to jump to the next row        
        i = i + 1
        if i == len(contours):
            print "Now we are there..."
        
        if i > n:
            print "Something is wrong"
    # When loop is done we return the matrix of features 
    print("Inside the function the featureMatrix is: ")
    print featuresMatrix

    # But to actually read it out we transpose the featureMatrix and return it.    
    featuresMatrix = featuresMatrix.T
    #print("The transposed version of featuresMatrix")
    #print featuresMatrix    
    return featuresMatrix    
    
##########################################q
# Main
##########################################

#video = 0
#video = 'DSC_0015.MOV'
#cap = cv2.VideoCapture(video)   

#For 1Simple.png and 2Simple.png... where 3 is not counted
featuresMatrix = []  
numberOfTrainingData = 10 # I.e. the number of images i use. So 1.png-9.pgn = 10
for i in range(1,numberOfTrainingData):      
    # "1.png" to "9.png" is training data, 
    #stringName = str(i) + "Simple.png"
    stringName = str(i) + ".png"
    print(stringName)
    img = cv2.imread(stringName, cv.CV_LOAD_IMAGE_COLOR)
    
    # Do the grayscale converting    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Do the thresholding    
    ret, threshold_img = cv2.threshold(grayImg,50,255,cv2.THRESH_BINARY)
    #cv2.imshow("Thresholded image", threshold_img)
    
    #Get the contours in the image. Actually no image processing required, but is 
    #done anyway to insure that the input image is 8UC1 - e.g. 1 channel 8 bit and not RGB.   
    contours = GetCountours(threshold_img)
    #print("The contours is: ", contours)
      
    #Analysing the contours
    temp = AnalyseContours(contours)
    featuresMatrix.append(temp)

print("......................\n")

area = []
perimeter = []
mu12 = []
for i in range(0,numberOfTrainingData-1):
   #print("i is: ", i)
   temp = featuresMatrix[i][0]
   area.append(temp)
#print("And finally the area is:", area)
#print("And the zero'th place in area contains:", area[0])

for j in range(0,numberOfTrainingData-1):
   #print("j is: ", j)
   temp = featuresMatrix[j][1]
   perimeter.append(temp)

for k in range(0,numberOfTrainingData-1):
   #print("j is: ", j)
   temp = featuresMatrix[k][2]
   mu12.append(temp)

#plt.plot(area[Image_N],perimeter[Image_N], 'X<o>', label='Number N')
plt.plot(area[0],perimeter[0], 'ro', label='Number 1')
plt.plot(area[1],perimeter[1], 'go', label='Number 2')
plt.plot(area[2],perimeter[2], 'bo', label='Number 3')
plt.plot(area[3],perimeter[3], 'co', label='Number 4')
plt.plot(area[4],perimeter[4], 'mo', label='Number 5')
plt.plot(area[5],perimeter[5], 'yo', label='Number 6')
plt.plot(area[6],perimeter[6], 'ko', label='Number 7')
plt.plot(area[7],perimeter[7], 'wo', label='Number 8')
plt.plot(area[8],perimeter[8], 'r*', label='Number 9')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=3, fancybox=True, shadow=True)
plt.ylabel("perimeter")
plt.xlabel("area")
plt.show()

plt.plot(mu12[0],perimeter[0], 'ro', label='Number 1')
plt.plot(mu12[1],perimeter[1], 'go', label='Number 2')
plt.plot(mu12[2],perimeter[2], 'bo', label='Number 3')
plt.plot(mu12[3],perimeter[3], 'co', label='Number 4')
plt.plot(mu12[4],perimeter[4], 'mo', label='Number 5')
plt.plot(mu12[5],perimeter[5], 'yo', label='Number 6')
plt.plot(mu12[6],perimeter[6], 'ko', label='Number 7')
plt.plot(mu12[7],perimeter[7], 'wo', label='Number 8')
plt.plot(mu12[8],perimeter[8], 'r*', label='Number 9')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=3, fancybox=True, shadow=True)
plt.ylabel("perimeter")
plt.xlabel("mu12")
plt.show()

# For the legend settings:
# http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot

#Now when the segmentation part is done, then it is time to define 
numberOfPerformanceData = 3
for i in range(1,numberOfPerformanceData):
    stringName = "0" + str(i) + ".png"
    print(stringName)
    img = cv2.imread(stringName, cv.CV_LOAD_IMAGE_COLOR)
    
    # Do the grayscale converting    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Do the thresholding    
    ret, threshold_img = cv2.threshold(grayImg,50,255,cv2.THRESH_BINARY)
    cv2.imshow("Thresholded image", threshold_img)
    
    while(1):
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print("User closed the program...")
            break  
#cap.release()
cv2.destroyAllWindows()

#print("Playing the video...")
#while(1):    
#    sucessfully_read, frame = cap.read()
#    if not sucessfully_read:
#        print("Video ended. Reloading video...")
#        cap.set(cv.CV_CAP_PROP_POS_AVI_RATIO, 0)        
#        continue
#    
#    #Resize the image by a given scale, 1 is 100%, 0.4 is 40%, etc. 
#    scale = 1
#    img_resize = ResizeImg(frame,scale)
#
#    # Do the grayscale converting    
#    grayImg = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)    
#    
#    # Do the thresholding  
#    ret, threshold_img = cv2.threshold(grayImg,50,255,cv2.THRESH_BINARY)
#
#    #Copy the image to an other image    
#    contourImg = threshold_img.copy()
#
#    #Get the contours in the image
#    contours = GetCountours(contourImg)
#    
#    #Analysing the contours
#    parameters = AnalyseContours(contours)   
#    
#
#    #Get the contours in the image
#   # contours = GetCountours(threshold_img)
#    
#    #cv2.imshow("Result of learning system", threshold_img)
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        print("User closed the program...")
#        break  
