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
def GetFeatures(contours, contourClass, areaThreshold):
    result = []    
    output = []    
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
    w = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
    b = random.uniform(0.0, 1.0)

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
    #Load the data in
    img = cv2.imread(string, cv2.CV_LOAD_IMAGE_COLOR)
    
    #Do the grayscale converting 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    
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
    
    #Draw the contours
    cv2.drawContours(img,contours,-1,(0,255,0),2)
    
    #Show the image
    cv2.imshow(string, img)
       
    #Return the contours
    return contours
    
def NormalizeData(trainingData):
    temp_area = []
    temp_compactness = []
    
    #Extract the area and compactness in traningData
    for index in trainingData:
        temp_area.append(index[0])
        temp_compactness.append(index[1])

    #Find the maximum value of area and compactness
    maxArea = max(temp_area)
    maxCompactness = max(temp_compactness)
        
    #print('-' * 60)
    
    #Normalize area and compactness in trainingData, so it form 0 to 1. 
    for index in trainingData:
        norm_area = index[0]/maxArea
        index[0] = norm_area
        
        norm_compactness = index[1]/maxCompactness
        index[1] = norm_compactness
        
    return trainingData

##########################################
# Libraries
##########################################
import numpy as np
import cv2
import matplotlib.pyplot as plt # required for plotting
import pylab  
import random
def main():
    
    #For each image, get the contours in the image
    contourTraining1 = GetContours("roundObjects.png")
    contourTraining2 = GetContours("squres_and_stuff.png")
    contourTesting = GetContours("testImage.png")
    
    print("The length of contourTraining1 is:%d" %(len(contourTraining1)))
    print("The length of contourTraining2 is:%d" %(len(contourTraining2)))
    print("The length of contourTesting is:%d" %(len(contourTesting)))
    
    # Find the features for each contours from each image
    # featureTraining1[[]] = [[area], [perimeter], [compactness]]
    areaThreshold = 10
    contourClass1 = 1
    contourClass2 = -1
    contourClass3 = 0
    
    featureTraining1 = GetFeatures(contourTraining1, contourClass1, areaThreshold)
    featureTraining2 = GetFeatures(contourTraining2, contourClass2, areaThreshold)
    featureTesting = GetFeatures(contourTesting, contourClass3, areaThreshold)
    
    #Extract the two features, area and compactness, form the two featureTraning data sets...
    area1 = []
    compactness1 = []
    for index in featureTraining1:
        area1.append(index[0])
        compactness1.append(index[1])
    
    area2 = []
    compactness2 = []
    for index in featureTraining2:
        area2.append(index[0])
        compactness2.append(index[1])
          
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
    trainingData = featureTraining1 + featureTraining2
    testingData = featureTesting    
    
    #Normalization of trainingData and testingData
    trainingData = NormalizeData(trainingData)    
    testingData =  NormalizeData(testingData)   
    
    #Extract again the two features, area and compactness, form the two featureTraning data sets...
    # that now has been normalized
    area_norm_training = []
    compactness_norm_training = []
    for index in trainingData:
        area_norm_training.append(index[0])
        compactness_norm_training.append(index[1])
     
    #Extract again the two features, area and compactness, form the two featureTesting data sets...
    # that now has been normalized
    area_norm_testing = []
    compactness_norm_testing = []
    for index in testingData:
        area_norm_testing.append(index[0])
        compactness_norm_testing.append(index[1])
    
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
    class1_area_list = []
    class1_compactness_list = []
    for index in class1:
        class1_area_list.append(index[0])
        class1_compactness_list.append(index[1])
    
    classNeg1_area_list = []
    classNeg1_compactness_list = []
    for index in classNeg1:
        classNeg1_area_list.append(index[0])
        classNeg1_compactness_list.append(index[1])
    
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
    
      
    while(1):
        #cv2.imshow("training1Img", training1Img)
        #cv2.imshow("trainin2Img", training2Img)
        #cv2.imshow("Test image", testImg)
        
        print 'I need to calculate the moment of each contour and calculate the x.y positon for each contour. Then I need to store this, so at the end I can pind out which figure was classified as what'        
        
        #cv2.imshow("trainin3Img", training3Img)    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print("User closed the program...")
            break  

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()