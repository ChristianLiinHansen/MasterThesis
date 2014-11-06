# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:29:21 2014

@author: christian
"""
##########################################
# Functions
##########################################
#The contourClass is an argument, since this function is used for the supervised
#learning.In SL we need to specify which class each contour belongs to. 
#If the contourClass belongs to 1, means the training data is i.e. circles
#else if the contourClass belongs to -1, means the training data is i.e. squares.
#else if the contourClass belongs to 0, means the data is not training data, but test data
def GetFeatures(contours, contourClass, areaThreshold, centers):
    #print ("Centers is:", centers)
    result = []    
    output = []
    i = 0
    #print("Length of centers is", len(centers))
    #print("Length of contours is", len(contours))    
    for contour in contours:
        #Get the area of each of the contours 
        temp_area = cv2.contourArea(contour,False)
    
        #Skip the iteration if the area is less than something     
        if temp_area < areaThreshold:
            continue
        
        #Get the perimeter of each of the contours
        temp_perimeter = cv2.arcLength(contour, 1); # 1 indicate that the contours is closed. 
        
        #Get the compactness of each of the contours        
        temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)
    
        #Append the area in result
        result.append(temp_area)
        
        #Append the compactness in result        
        result.append(temp_compactness) 
        
        #Append the which class the contour has in result
        result.append(contourClass)
        
        #Append the centroid coordinate for each contour
        result.append(centers[i])

        #Increment i to get to the next centroid.       
        i+=1
        #print ("The center of this contour is: %d" %(centers[]))
        #print 'he center of this contour is: {}"' .format(centers[contour])
        #print "The contours center is: %d and the area is: %d and compactness is %d" %(centers, temp_area, temp_compactness)
        
        #Store the information for each contour in the output list.
        # output = [[area0, compactness0, contourClass0], [area1, compactness1, contourClass1],...]
        output.append(result)
        
        #Clear the result list for each contour        
        result = []    
    
    #When loop is done, the output is returned    
    return output

#trainingData = [[area, compactness, contourClass], [area, compactness, contourClass],...]
def Perceptron(trainingData, learning_rate):
    print("Now the perceptron starts")    
    
    output = []
    error_plot = []
    w0_plot = []
    w1_plot = []
    b_plot = []
    
    #Initial random weights and bias from 0.0 to 1.0
    #w = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
    #b = random.uniform(0.0, 1.0)
    
    w = [0.00001, 0.00001]
    b = 0.00001
    #Start the algorithm
    runFlag = True
    trueCounter = 0
    while runFlag == True:
        trueCounter += 1
        #print('-' * 60)
        error_count = 0      
        
        for data in trainingData:
            #print("The weights is:", w)            
            #Calculate the dotproduct between input and weights            
            dot_product = data[0]*w[0] + data[1]*w[1]
            
            #If the dotprodcuct + the bias is >= 0, then result is class 1
            # else it is class -1. 
            if dot_product + b >= 0:
                result = 1
            else:
                result = -1
            
            #Calculate error, where data[2] is the contourClass/desired output 
            error = data[2] - result

            #Continue the while, continue the algorithm if only the error is not zero
            if error != 0:
                error_count += 1
                #Update the final waits and bias
                w[0] += data[0]*learning_rate*error
                w[1] += data[1]*learning_rate*error
                b += learning_rate * error
        
            #Store the weights and bias
            w0_plot.append(w[0])
            w1_plot.append(w[1])
            b_plot.append(b)
            error_plot.append(error)
    
        if error_count == 0:
           # print("Now there is no errors in the whole trainingData")
            runFlag = False
    print("The number of iterations before the Perceptron stops is:", trueCounter)
    
    plt.figure("Plot weights, bias and error")
    plt.title("Plot weights, bias and error")
    #plt.xlabel('Area')
    #plt.ylabel('w0')
    plt.plot(w0_plot, 'b-', label="w0")
    plt.plot(w1_plot, 'g-', label="w1")
    plt.plot(b_plot, 'r-', label="b")
    plt.plot(error_plot, 'c-', label="error")
#    ax = plt.subplot(111)
#    ax.legend(loc='lower center', bbox_to_anchor=(0.8, 0.9),
#          ncol=3, fancybox=True, shadow=True)
    plt.legend(bbox_to_anchor=(0.0, 1.1))
    plt.show(block = False)
    
    output.append(w)
    output.append(b)
    return output

def GetContours(string):
    #Do the grayscale converting 
    gray_img = cv2.cvtColor(string, cv2.COLOR_BGR2GRAY)    
    
    # Show the thresholding
    #cv2.imshow("Grayscale image", gray_img)     
    
    #Do the thresholding
    ret,threshold_img = cv2.threshold(gray_img,250,255,cv2.THRESH_BINARY_INV)
    
    # Show the thresholding
    #cv2.imshow("New image", threshold_img)    
    
    #Copy the image, to avoid manipulating with original
    contour_img = threshold_img.copy()
    
    #Find the contours of the thresholded image
    contours, hierarchy = cv2.findContours(contour_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
    
    #Show the image
    #cv2.imshow(string, img)
       
    #Return the contours
    return contours

def GetCentroid(contours):
    
    centers = []
    
    #Run through all the contours    
    for contour in contours:
        
        #Calculate the moments for each contour in contours
        m = cv2.moments(contour)
        
        #If somehow one of the moments is zero, then we brake and reenter the loop (continue)
        #to avoid dividing with zero
        if (int(m['m01']) == 0 or int(m['m00'] == 0)):
            continue
        
        #Calculate the centroid x,y, coordinate out from standard formula.         
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        
        #Append each calculated center into the centers list.        
        centers.append(center)
    return centers    

def DrawCentroid(img, centers, RGB_list):
    # Color the central coordinates for red bricks with a filled circle
    for center in centers:
        cv2.circle(img, center, 5, RGB_list, -1)
    
def NormalizeData(data, maxValueArea, maxValueCompactness):
#    temp_area = []
#    temp_compactness = []
#    
#    #Extract the area and compactness in data
#    for index in data:
#        temp_area.append(index[0])
#        temp_compactness.append(index[1])

    #Find the maximum value of area and compactness
#    maxArea = max(temp_area)
#    maxCompactness = max(temp_compactness)
#        
    #Normalize area and compactness in data, so it form 0 to 1. 
    for index in data:
        norm_area = index[0]/maxValueArea
        index[0] = norm_area
        
        norm_compactness = index[1]/maxValueCompactness
        index[1] = norm_compactness
        
    return data
   
def Extract(inputList, element):
    outputList = []
    for eachList in inputList:
        outputList.append(eachList[element])  
    return outputList

def FindMaxValueOfLists(list1, list2, element):
    temp = []
    for index in list1:
        temp.append(index[element])
        
    #Find the maximum value of area
    maxValue1 = max(temp)

    for index in list2:
        temp.append(index[element])    
    
    maxValue2 = max(temp)
    maxValue = (maxValue1 + maxValue2)/2
    return maxValue
##########################################
# Libraries
##########################################
import numpy as np
import cv2
import matplotlib.pyplot as plt # required for plotting
import pylab  
import random
def main():
    
    #Load the data in
    img1 = cv2.imread("roundObjects.png", cv2.CV_LOAD_IMAGE_COLOR)
    img2 = cv2.imread("squres_and_stuff.png", cv2.CV_LOAD_IMAGE_COLOR)
    imgTest = cv2.imread("testImage.png", cv2.CV_LOAD_IMAGE_COLOR)
    
    #For each image, get the contours in the image
    contourTraining1 = GetContours(img1)
    contourTraining2 = GetContours(img2)
    contourTesting = GetContours(imgTest)
    
    #Get the central mass coordinate of each contour - used to illustrated each tracked object 
    # in the end of the program
    centers1 = GetCentroid(contourTraining1)
    centers2 = GetCentroid(contourTraining2)
    centersTesting = GetCentroid(contourTesting)
    
    print("The length of contourTraining1 is:%d" %(len(contourTraining1)))
    print("The length of contourTraining2 is:%d" %(len(contourTraining2)))
    print("The length of contourTesting is:%d" %(len(contourTesting)))
    
    # Find the features for each contours from each image
    # featureTraining1[[]] = [[area], [perimeter], [compactness]]
    areaThreshold = 10
    contourClass1 = 1
    contourClass2 = -1
    contourClass3 = 0
    
    featureTraining1 = GetFeatures(contourTraining1, contourClass1, areaThreshold, centers1)
    featureTraining2 = GetFeatures(contourTraining2, contourClass2, areaThreshold, centers2)
    featureTesting = GetFeatures(contourTesting, contourClass3, areaThreshold, centersTesting)
    

    #Extract the two features, area and compactness, form the two featureTraning data sets...
    area1 = Extract(featureTraining1, 0)    
    compactness1 = Extract(featureTraining1, 1)
    
    area2 = Extract(featureTraining2, 0)    
    compactness2 = Extract(featureTraining2, 1) 
              
    #Draw the featuers from the training data set
    plt.figure("Feature space for training")
    plt.title("Feature space for training")
    plt.plot(area1,compactness1, 'ro', label = "circles")
    plt.plot(np.mean(area1), np.mean(compactness1), 'rs', markersize=20)
    plt.plot(area2,compactness2, 'bs', label = "rectangles")
    plt.plot(np.mean(area2), np.mean(compactness2), 'bs', markersize=20)
    plt.legend(bbox_to_anchor=(1.0, 1.15))
    plt.grid(True)
    plt.xlabel('Area')
    plt.ylabel('Compactness')
    #plt.xlim(0,1)
    plt.ylim(0.5,1)
    plt.show(block = False)

    #Add the training data together
    print featureTraining1
    trainingData = featureTraining1 + featureTraining2
    testingData = featureTesting

    #Before finding the maximum value of both feature list
    maxValueArea = FindMaxValueOfLists(trainingData, testingData, 0)
    maxValueCompactness = FindMaxValueOfLists(trainingData, testingData, 1)     
    
    #Normalization of trainingData and testingData
    trainingData = NormalizeData(trainingData, maxValueArea, maxValueCompactness)    
    testingData =  NormalizeData(testingData, maxValueArea, maxValueCompactness)   
    
    #Extract again the two features, area and compactness, form the two featureTraning data sets...
    # that now has been normalized    
    area_norm_training = Extract(trainingData, 0)    
    compactness_norm_training = Extract(trainingData, 1) 

    #Extract again the two features, area and compactness, form the two featureTesting data sets...
    # that now has been normalized
    area_norm_testing = Extract(testingData, 0)    
    compactness_norm_testing = Extract(testingData, 1)    
    
    #Draw the featuers from the training data set - normalized
#    plt.figure("Feature space for training - normalized")
#    plt.title("Feature space for training - normalized")
#    plt.plot(area_norm_training, compactness_norm_training, 'g*', label = "Normalized traning data")
#    plt.legend(bbox_to_anchor=(1.0, 1.1))
#    plt.grid(True)
#    plt.xlabel('Area')
#    plt.ylabel('Compactness')
#    plt.show(block = False)
    
    #Now the area and compactness in trainingData has been normalized    
    
    #Run the Perceptron algorithm to learn the classifier something...
    learning_rate = 0.10
    result = []
    
    #Shuffle the trainingData before sending the data into the Perceptron
    #print ("Before shuffle", trainingData)
    random.shuffle(trainingData)
    #print("And the shuffle is:",trainingData)
    
    #The result[0] is the final weights and result[1] is the final bias 
    result = Perceptron(trainingData, learning_rate)
    print("Now perceptron is done")
    print("And the result is:", result)
    
    w = result[0]
    b = result[1]
    
    #Draw the classifier
    wx = pylab.arange(0,1,0.01)
    wy = (w[0]*wx)/(-w[1]) + (b)/(-w[1])

    plt.figure("Feature space with classifier seperater - The perceptron")
    plt.title("Feature space with classifier seperater - The perceptron")
    plt.plot(wx,wy, 'b-', label = "The perceptron line")
    plt.plot(area_norm_training, compactness_norm_training, 'g*', label = "Normalized training data")
    plt.legend(bbox_to_anchor=(1.0, 1.15))
    plt.grid(True)
    plt.xlabel('Area')
    plt.ylabel('Compactness')
    plt.xlim(0,1)
    plt.ylim(0.5,1)    
    plt.show(block = False)
    
    #With the ready Perceptron classifier, we can now classify the testing data
    # and mark that on the original testing image.
    
    #Doing the classification. So if the y is negative, it belongs to class -1
    # and if the y is positive it belongs to class 1. 
    # Before the testingData is intered the classifier, the data[2] = 0 --> unclassified.
    # After this for loop the data[2] is either -1 or +1
    class1 = []
    classNeg1 = []    
    for index in testingData:
        y = index[0]*w[0] + index[1]*w[1] + b
        if y >= 0:
            index[2] = 1
            class1.append(index)
        else:
            index[2] = -1
            classNeg1.append(index)
    
    #Extract the two features, area and compactness, form the testing data sets...
    # that has been normalized
    class1_area_list = Extract(class1, 0)    
    class1_compactness_list = Extract(class1, 1)
    classNeg1_area_list = Extract(classNeg1, 0)    
    classNeg1_compactness_list = Extract(classNeg1, 1)   
    
    plt.figure("test")
    plt.title("test")
    plt.plot(wx,wy, 'b-', label = "The perceptron line")
    plt.plot(class1_area_list, class1_compactness_list, 'ro', label = "Class1 classified data")
    plt.plot(classNeg1_area_list, classNeg1_compactness_list, 'bo', label = "ClassNeg1 classified data")
    plt.legend(bbox_to_anchor=(1.0, 1.15))    
    plt.grid(True)
    plt.xlabel('Area')
    plt.ylabel('Compactness')
    plt.xlim(0,1)
    plt.ylim(0.7,1)    
    plt.show(block = False)      
    
    #Draw the featuers from the testing data set
    plt.figure("Feature space for testing")
    plt.title("Feature space for testing")
    plt.plot(wx,wy, 'b-', label = "The perceptron line")    
    plt.plot(area_norm_testing, compactness_norm_testing, 'go', label = "Normalized testing data")    
    plt.legend(bbox_to_anchor=(1.0, 1.15))
    plt.grid(True)    
    plt.xlabel('Area')
    plt.ylabel('Compactness')
    plt.xlim(0,1)
    plt.ylim(0.5,1)
    plt.show(block = False)
    
    #Draw the contours
    cv2.drawContours(img1,contourTraining1,-1,(0,255,0),2)
    cv2.drawContours(img2,contourTraining2,-1,(0,255,0),2)
    cv2.drawContours(imgTest,contourTesting,-1,(0,255,0),2)         
    
    #We define circles to be red, and squares to be blue
    DrawCentroid(img1,centers1,(0,0,255))
    DrawCentroid(img2,centers2,(255, 0,0))
    
    #centers = []
    #centers = Extract(testingData, 3)
    #print centers        
    
    #Do something like this:
    for index in testingData:
        if(index[2] == -1):
            cv2.circle(imgTest, index[3], 5, (255,0,0), -1)
        elif(index[2] == 1):
            cv2.circle(imgTest, index[3], 5, (0,0,255), -1)             
        else:
            print("Should not come into this else")
      
    cv2.imshow("training1Img", img1)
    cv2.imshow("training2Img", img2)     
    cv2.imshow("trainingTest", imgTest)     
    
    while(1):
        #cv2.imshow("trainin3Img", training3Img)    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print("User closed the program...")
            break  

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()