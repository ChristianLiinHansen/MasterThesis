# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:29:21 2014

@author: christian
"""
##########################################
# Functions
##########################################
def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))

def GetFeatures(contours):
    area = []
    perimeter = []
    compactness = []
    output_vector = []

    for contour in contours:
        #Get the area, perimeter and compactness of the contours in contours1    
        temp_area = cv2.contourArea(contour,False)
    
        #Skip the iteration if the area is less than something     
        if temp_area < 10:
            continue 
        temp_perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed. 
        temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)
    
        area.append(temp_area)   
        perimeter.append(temp_perimeter)
        compactness.append(temp_compactness)
    
    output_vector.append(area)
    output_vector.append(perimeter)
    output_vector.append(compactness) 
    return output_vector

##########################################
# Libraries
##########################################
import numpy as np
import cv2
import matplotlib.pyplot as plt # required for plotting

#Load the data in
training1Img = cv2.imread("roundObjects.png", cv2.CV_LOAD_IMAGE_COLOR)
training2Img = cv2.imread("Squres_and_stuff.png", cv2.CV_LOAD_IMAGE_COLOR)
#training3Img = cv2.imread("triangles.png", cv2.CV_LOAD_IMAGE_COLOR)
testImg = cv2.imread("testImage.png", cv2.CV_LOAD_IMAGE_COLOR)

# Do the grayscale converting   
grayImg1 = cv2.cvtColor(training1Img, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(training2Img, cv2.COLOR_BGR2GRAY) 
#grayImg3 = cv2.cvtColor(training3Img, cv2.COLOR_BGR2GRAY)
grayTest = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)

#Do the thresholding
ret,thresh1 = cv2.threshold(grayImg1,220,255,cv2.THRESH_BINARY_INV)
ret,thresh2 = cv2.threshold(grayImg2,220,255,cv2.THRESH_BINARY_INV)
#ret,thresh3 = cv2.threshold(grayImg3,220,255,cv2.THRESH_BINARY_INV)
ret,threshTest = cv2.threshold(grayTest,220,255,cv2.THRESH_BINARY_INV)

# Contour filling, maybee?

# Finding the contours
# Before applying the findContours, we need to clone the image, otherwise
# we mess with the original image
contourImage1 = thresh1.copy()
contourImage2 = thresh2.copy()
#contourImage3 = thresh3.copy()
contourImageTest = threshTest.copy()

contours1, hierarchy = cv2.findContours(contourImage1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
contours2, hierarchy = cv2.findContours(contourImage2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
#contours3, hierarchy = cv2.findContours(contourImage3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
contoursTest, hierarchy = cv2.findContours(contourImageTest,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Draw the contours
cv2.drawContours(training1Img,contours1,-1,(0,255,0),2)
cv2.drawContours(training2Img,contours2,-1,(0,255,0),2)
#cv2.drawContours(training3Img,contours3,-1,(0,255,0),2)
cv2.drawContours(threshTest,contoursTest,-1,(0,255,0),2)

#Create the list for each image
area1 = []
perimeter1 = []
compactness1 = []

area2 = []
perimeter2 = []
compactness2 = []

#area3 = []
#perimeter3 = []
#compactness3 = []

areaTest = []
perimeterTest = []
compactnessTest = []

test = [[]]
test = GetFeatures(contours1)

print("Now the testing begings...")
print("The result is", test)

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

#for contour in contours3:
#    #Get the area, perimeter and compactness of the contours in contours1    
#    temp_area = cv2.contourArea(contour,False) 
#    
#    #Skip the iteration if the area is less than something 
#    if temp_area < 10:
#        continue 
#    
#    temp_perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed. 
#    temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)
#    
#    area3.append(temp_area)   
#    perimeter3.append(temp_perimeter)
#    compactness3.append(temp_compactness)

for contour in contoursTest:
    #Get the area, perimeter and compactness of the contours in contours1    
    temp_area = cv2.contourArea(contour,False) 
    
    #Skip the iteration if the area is less than something 
    if temp_area < 10:
        continue     
    
    temp_perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed. 
    temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)
    
    areaTest.append(temp_area)   
    perimeterTest.append(temp_perimeter)
    compactnessTest.append(temp_compactness)
    
#So now we have tree features from tree images
# Let plot the featuers in a 2D plot.
plt.figure(1)
plt.plot(area1, perimeter1, 'ro')
plt.plot(area2, perimeter2, 'go')
#plt.plot(area3, perimeter3, 'bo')

plt.title('Feature space')
plt.xlabel('Area')
plt.ylabel('Perimeter')

plt.figure(2)
plt.plot(area1, compactness1, 'ro')
plt.plot(area2, compactness2, 'go')
#plt.plot(area3, compactness3, 'bo')

#Plot the mean of the cluster
plt.plot(np.mean(area1), np.mean(compactness1), 'rs', markersize=20)
#Plot the mean of the cluster
plt.plot(np.mean(area2), np.mean(compactness2), 'gs', markersize=20)
#Plot the mean of the cluster
#plt.plot(np.mean(area3), np.mean(compactness3), 'bs', markersize=20)

plt.title('Feature space')
plt.xlabel('Area')
plt.ylabel('Compactness')

plt.figure(3)
plt.plot(perimeter1, compactness1, 'ro')
plt.plot(perimeter2, compactness2, 'go')
#plt.plot(perimeter3, compactness3, 'bo')
plt.title('Feature space')
plt.xlabel('Perimeter')
plt.ylabel('Compactness')

#And then show all the figures
#plt.show()



#Now the idea is to implement a simply linear classifier, using
#the Perceptron. In this case, one feature is simply enough to 
#seperate the data into two clusters. 
# When the examples with i.e. numbers in the SUDOKU is implemented
# then the neurale network should have more than 1 input.

#But in this simple case we use the most simple ANN, the perceptron.
# One input, x, one weight w, and one output y. 
#
#   x --w-->(Neuron) --> y
#

#print("The compactnes for training data set 1 is:", compactness1)
#print('-' * 60)
#print("The compactnes for training data set 2 is:", compactness2)
#print('-' * 60)
#print("To concatunate two vector is done simple as is:", compactness1 + compactness2 )
#To shuffle them randomly, we do this:
    
#compactness_list = compactness1 + compactness2
#random.shuffle(compactness_list)
#print("Now the list is combined and shufled", compactness_list)

#testA = [1,2,3,4,5,6,7,8,9]
#random.shuffle(testA)
#print("And the shuffle is:",testA)
#print("The first element is:", compactness_list[0])

# Use the zip function to make a set. 
# Like compactness1 = [c0,c1,...cn] and area1 = [a0,a1,a2....]
# so with zip we get [(c0,a0), (c1,a1),....,(cn,an)]




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

# Input
x = 0.0
# Weight
w0 = 0.0
w1 = 0.0
# Unclear if the if the w should be a np.array or list 
# Output
y = 0.0
#Learning reate r
r = 0.1
#Threshold
t = 0.7
#Error
e = 0.0

# Desired output #Seen from the training data. Could be calculated directly when 
# having the other class. 
# Like: upper class min = 0.90 and lower class max = 0.80, then threshold is
# (min+max)/2 = 0.85

compactness_threshold = 0.85
error_plot = []
w0_plot = []
w1_plot = []

# First we need to normalized the area of image1
array1 = np.array(area1)
array2 = np.array(area2)

maxValue1 = max(array1)
maxValue2 = max(array2)

norm_array1 = array1/maxValue1
norm_array2 = array1/maxValue2

zipped1 = zip(compactness1, norm_array1)
zipped2 = zip(compactness2, norm_array2)

#print("Length of zipped1 is:",len(zipped1))
#print("Length of zipped2 is:",len(zipped2))
zipped = zipped1 + zipped2
#print("Length of zipped is:",len(zipped))

#Before sending in the data, we shuffle it ...
#print("Before the shuffle the vector is: ",zipped)
#random.shuffle(zipped)
#print("After the shuffle the vector is: :",zipped)

for input_vector in zipped:
    print("The input_vector contains (compactness, area):", input_vector)
    print('-' * 60)
    
    #Define the x0 and x1 out from the input vector. x0 is compactness and x1 is area
    x0 = input_vector[0]
    x1 = input_vector[1]
    
    #Find the desired output z    
    if(x0 > compactness_threshold):
        z = 1
        print 'Now z = 1'
    else:
        z = 0
        print 'Now z = 0'
    
    #Show the weigts
    print("w0 is: ", w0)
    print("w0 is: ", w1)
    
    #Calclatethe c
    c0 = x0*w0
    c1 = x1*w1

    # Calulate the sum
    s = c0 + c1

    # Calulate the n
    if(s > compactness_threshold):
        n = 1
    else:
        n = 0

    #Calculate error
    e = z - n
    error_plot.append(e)

    #Calculate correction
    d = r*e

    #Update the final waits
    w0 = w0 + x0*d
    w1 = w1 + x1*d
    
    w0_plot.append(w0)
    w1_plot.append(w1)
    
    #Print the error
    print("The error is: ", e)

# Now the perceptron has learned... (when the error is 0 for a long time)
print('-' * 60)
print 'After learning'

plt.figure()
plt.plot(error_plot, 'r', markersize=5)
plt.plot(w0_plot, 'g', markersize=5)
plt.plot(w1_plot, 'b', markersize=5)
plt.show()
#Plot the mean of the cluster
#plt.plot(np.mean(area2), np.mean(compactness2), 'gs', markersize=20)
#Plot the mean of the cluster
#plt.plot(np.mean(area3), np.mean(compactness3), 'bs', markersize=20)

#Find the contours for the final test image
#Calculate the output.
print ("The final weights after learning is w0: ", w0, "and w1: ", w1)

#y = data_input0 * w0 + data_input1 * w1
#
#if (y > 0)
#data --> til den ene side
#else (y < 0)
#
#print y

# Dette var rigtig tænkt! Blot udregn compactness og area for testing data og ikke længere
# træning data. Derefter sæt ind i i y og sige: Hvis y > 0 er det classe 1
# og ellers hvis y < 0 er det classe 2. 
# Husk også at tilføje bias, således at man kan seperere flere lineære problemer.
# Ellers kan vi kun seperere ting der går gennem 0,0, hvis bias  = 0.



#Somewher here I need to find the contour of the test and get the 
# compactness and area out. 
#Then define the new data_vector as [(c0,a0), (c1,a1), ..., (cn,an)] wher
# the cn and an is talking from the testing image. 

# Then apply the neron, so Sum = datainput[0]*w0 + datainput[1]*w1
# if Sum > threshold ---> y = 1, else y = 0.
# If y == 1, then the contour is a circle and else if y == 0 then the contour is square. 


  
#    if (e == 0):
#        break
      
while(1):
    cv2.imshow("training1Img", training1Img)
    cv2.imshow("trainin2Img", training2Img)
    cv2.imshow("Test image", testImg)
     
    
    #cv2.imshow("trainin3Img", training3Img)    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("User closed the program...")
        break  

cv2.destroyAllWindows()