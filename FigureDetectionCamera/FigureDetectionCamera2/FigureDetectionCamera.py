# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:29:21 2014

@author: christian

Website inspirations:
http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
"""

##########################################
# Functions
##########################################
#The contourClass is an argument, since this function is used for the supervised
#learning.In SL we need to specify which class each contour belongs to.
#If the contourClass belongs to 1, means the training data is i.e. circles
#else if the contourClass belongs to -1, means the training data is i.e. squares.
#else if the contourClass belongs to 0, means the data is not training data, but test data

##########################################
# Libraries
##########################################
import numpy as np                  # required for calculate i.e mean with np.mean
import cv2                          # required for use OpenCV
import matplotlib.pyplot as plt     # required for plotting
import pylab                        # required for arrange doing the wx list
import random                       # required to choose random initial weights and shuffle data

##########################################
# Classes
##########################################
class ProcessImage(object):

    #The constructor will run each time an object is assigned to this class.
    def __init__(self, image, class_name):
        self.img = image
        self.getGrayImg()
        self.thresholding(250)
        self.contours = self.findContours()
        self.centers = self.getCentroid(self.contours)
        self.features = self.getFeatures(class_name, 10)
        self.area = self.extract(self.features, 0)
        self.compactness = self.extract(self.features, 1)
        self.class_name = self.extract(self.features, 2)

    def showImage(self, window_name):
        cv2.imshow(window_name, self.img)

    def getGrayImg(self):
        self.grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Grayscale image", self.grayImg)

    def showGrayImg(self, window_name):
        cv2.imshow(window_name, self.grayImg)

    def thresholding(self, threshold):
        self.ret, self.threshold_img = cv2.threshold(self.grayImg, threshold, 255, cv2.THRESH_BINARY_INV)

    def showThresholding(self, window_name):
        cv2.imshow(window_name, self.threshold_img)

    def findContours(self):
        self.contourImg = self.threshold_img.copy()
        self.contours, self.hierarchy = cv2.findContours(self.contourImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return self.contours

    def getCentroid(self, contours):
        centers = []
        #Run through all the contours
        for contour in contours:

            #Calculate the moments for each contour in contours
            m = cv2.moments(contour)

            #If somehow one of the moments is zero, then we brake and reenter the loop (continue)
            #to avoid dividing with zero
            if int(m['m01']) == 0 or int(m['m00'] == 0):
                continue

            #Calculate the centroid x,y, coordinate out from standard formula.
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

            #Append each calculated center into the centers list.
            centers.append(center)
        return centers

    def getFeatures(self, contour_class, area_threshold):
        result = []
        output = []
        i = 0
        for contour in self.contours:
            #Get the area of each of the contours
            temp_area = cv2.contourArea(contour, False)

            #Skip the iteration if the area is less than something
            if temp_area < area_threshold:
                continue

            #Get the perimeter of each of the contours
            temp_perimeter = cv2.arcLength(contour, 1) # 1 indicate that the contours is closed.

            #Get the compactness of each of the contours
            temp_compactness = (4 * 3.141592 * temp_area) / (temp_perimeter * temp_perimeter)

            #Append the area in result
            result.append(temp_area)

            #Append the compactness in result
            result.append(temp_compactness)

            #Append the which class the contour has in result
            result.append(contour_class)

            #Append the centroid coordinate for each contour
            result.append(self.centers[i])

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

    def drawCenters(self, centers, rgb_list):
        # Color the central coordinates for red bricks with a filled circle
        for center in centers:
            cv2.circle(self.img, center, 5, rgb_list, -1)

    def extract(self, inputList, element):
        outputList = []
        for eachList in inputList:
            outputList.append(eachList[element])
        return outputList

class PlotFigures():
    def __init__(self, string):
        self.name = string
        self.fig = plt.figure(self.name)
        plt.title(self.name)
        self.ax = plt.subplot(111)

    def plotData(self, x, y, string_icon, string_label):
        plt.plot(x, y, string_icon, label=string_label)

        # Shrink current axis by 20%
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # Put a legend to the right of the current axis
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #Set grid on, limit the y axis (not the x yet) and put names on axis
        plt.grid(True)

    def setXlabel(self, string_x):
        plt.xlabel(string_x)

    def setYlabel(self, string_y):
        plt.ylabel(string_y)

    def limit_y(self, min_y, max_y):
        plt.ylim(min_y, max_y)

    def limit_x(self, min_x, max_x):
        plt.xlim(min_x, max_x)

    def plotMean(self, x, y, string_icon):
        plt.plot(np.mean(x), np.mean(y), string_icon, markersize=20)

    def updateFigure(self):
        plt.show(block=False)

class Normalize(ProcessImage):
    # The class Normalize is a subclass from the ProcessImage class.
    # In that way the object that is instantiated from Normalize class has
    # access to the methods that lies within the ProcessImage class.
    # E.g. the extraction methods is not defined in Normalize class, but
    # the object that is instantiated from this class can find the extraction method
    # anyway.

    def __init__(self, list1, list2):
        #Finding the maximum values of features
        self.max_area = self.findMaxValueOfLists(list1, list2, 0)
        self.max_compactness = self.findMaxValueOfLists(list1, list2, 1)

        #Normalize the training and testing data
        self.training_data_normed = self.normalizeData(list1)
        self.testing_data_normed = self.normalizeData(list2)

        #Extract area and compactness from the training set after they been normalized
        self.area_norm_training = self.extract(self.training_data_normed, 0)
        self.compactness_norm_training = self.extract(self.training_data_normed, 1)

        #Extract area and compactness from the testing set after they been normalized
        self.area_norm_testing = self.extract(self.testing_data_normed, 0)
        self.compactness_norm_testing = self.extract(self.testing_data_normed, 1)

    def findMaxValueOfLists(self, list1, list2, element):
        temp = []
        for index in list1:
            temp.append(index[element])

        #Find the maximum value of area
        max_value1 = max(temp)

        #Clear the temp list
        # temp = []

        for index in list2:
            temp.append(index[element])

        max_value2 = max(temp)
        max_value = (max_value1 + max_value2)/2
        return max_value

    def normalizeData(self, data):
        for index in data:
            norm_area = index[0]/self.max_area
            index[0] = norm_area
            norm_compactness = index[1]/self.max_compactness
            index[1] = norm_compactness
        return data

class Perceptron(ProcessImage):
    def __init__(self, total_training_data):
        self.training_data = total_training_data
        random.shuffle(self.training_data)

        #Initial random weights and bias from 0.0 to 1.0
        #self.w = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
        #self.b = random.uniform(0.0, 1.0)

        self.w = [0.00001, 0.00001]
        self.b = 0.00001
        self.error = 0
        self.run_flag = True
        self.true_counter = 0

        # For plotting
        self.w0_plot = []
        self.w1_plot = []
        self.b_plot = []
        self.error_plot = []

        self.wx = 0
        self.wy = 0

        # For the classificaton
        self.class1 = []
        self.classNeg1 = []
        self.class1_area = []
        self.class1_compactness = []
        self.classNeg1_area = []
        self.classNeg1_compactness = []

    def startLearn(self, learning_rate):
        print("Now the perceptron starts")

        #Start the algorithm. RunFlag is already True
        self.true_counter = 0
        while self.run_flag == True:
            self.true_counter += 1
            #print('-' * 60)
            error_count = 0

            for data in self.training_data:
                #print("The weights is:", w)
                #Calculate the dotproduct between input and weights
                dot_product = data[0]*self.w[0] + data[1]*self.w[1]

                #If the dotprodcuct + the bias is >= 0, then result is class 1
                # else it is class -1.
                if dot_product + self.b >= 0:
                    result = 1
                else:
                    result = -1

                #Calculate error, where data[2] is the contourClass/desired output
                self.error = data[2] - result

                #Continue the while, continue the algorithm if only the error is not zero
                if self.error != 0:
                    error_count += 1
                    #Update the final waits and bias
                    self.w[0] += data[0]*learning_rate*self.error
                    self.w[1] += data[1]*learning_rate*self.error
                    self.b += learning_rate * self.error

                #Store the weights and bias
                self.w0_plot.append(self.w[0])
                self.w1_plot.append(self.w[1])
                self.b_plot.append(self.b)
                self.error_plot.append(self.error)

            if error_count == 0:
                # print("Now there is no errors in the whole trainingData")
                self.run_flag = False
        print("The number of iterations before the Perceptron stops is:", self.true_counter)

    def getClassifier(self):
        self.wx = pylab.arange(0, 1, 0.01)
        self.wy = (self.w[0]*self.wx)/(-self.w[1]) + (self.b)/(-self.w[1])

    def doClassification(self, testingData, finalImage):
    #With the ready Perceptron classifier, we can now classify the testing data
    # and mark that on the original testing image.

    #Doing the classification. So if the y is negative, it belongs to class -1
    # and if the y is positive it belongs to class 1.
    # Before the testingData is intered the classifier, the data[2] = 0 --> unclassified.
    # After this for loop the data[2] is either -1 or +1
        for index in testingData:
            y = index[0]*self.w[0] + index[1]*self.w[1] + self.b
            if y >= 0:
                index[2] = 1
                self.class1.append(index)
            else:
                index[2] = -1
                self.classNeg1.append(index)

        # Extract the area and compactness for each class
        self.class1_area = self.extract(self.class1, 0)
        self.class1_compactness = self.extract(self.class1, 1)
        self.classNeg1_area = self.extract(self.classNeg1, 0)
        self.classNeg1_compactness = self.extract(self.classNeg1, 1)

        for index in testingData:
            if(index[2] == -1):
                cv2.circle(finalImage, index[3], 5, (255, 0, 0), -1)
            elif(index[2] == 1):
                cv2.circle(finalImage, index[3], 5, (0, 0, 255), -1)
            else:
                print("Should not come into this else")

class ProcessVideo(ProcessImage):
        #The constructor will run each time an object is assigned to this class.
    def __init__(self, string):
        self.cap = cv2.VideoCapture(string)
        self.frame = 0

    def getFrame(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            return self.frame
        else:
            print 'Cant open video'

def main():

    # Training data 1. Define round objects as class 1
    image1 = cv2.imread("roundObjects.png", cv2.CV_LOAD_IMAGE_COLOR)
    td1 = ProcessImage(image1, 1)
    td1.drawCenters(td1.centers, (0, 0, 255))
    td1.showImage("The round objects where contours has been drawn")

    # Training data 2. Define round objects as class -1
    image2 = cv2.imread("squres_and_stuff.png", cv2.CV_LOAD_IMAGE_COLOR)
    td2 = ProcessImage(image2, -1)
    td2.drawCenters(td2.centers, (255, 0, 0))
    td2.showImage("The rectangles objects where contours has been drawn")

    # Now with testing data, which comes from a camera
    video = ProcessVideo(0)

    testingVideo = video.getFrame()
    cv2.imshow("Test of video", testingVideo)

    # Testing data. Define round objects as class 0
    testImage = cv2.imread("testImage.png", cv2.CV_LOAD_IMAGE_COLOR)
    testData = ProcessImage(testingVideo, 0)
    testData.showImage("The testing image with mixture of objects where contours has been drawn")


    #Draw data
    drawData1 = PlotFigures("Feature space for training data")
    drawData1.plotData(td1.area, td1.compactness, "rs", "circles")
    drawData1.setXlabel('Area')
    drawData1.setYlabel('Compactness')
    drawData1.plotData(td2.area, td2.compactness, "bs", "rectangles")
    drawData1.updateFigure()

    #Add the training data
    total_training_data = td1.features + td2.features
    print "The training1 data output is:", td1.features
    print "The training2 data output is:", td2.features
    print "The testing data output is:", testData.features

    # Use the Normalize class to normalize data
    normData = Normalize(total_training_data, testData.features)

    # The Perceptron takes care of shuffleing the training data
    p = Perceptron(normData.training_data_normed)

    # Start the Perceptron, with learning rate as argument
    p.startLearn(0.10)

    #Extract the classifier
    p.getClassifier()

    #Draw result of the Perceptron classifier
    perceptronResult = PlotFigures("Plot weights, bias and error")
    perceptronResult.plotData(range(0, len(p.w0_plot)), p.w0_plot, "b-", "w0")
    perceptronResult.setXlabel('Iterations')
    perceptronResult.setYlabel('Value')
    perceptronResult.plotData(range(0, len(p.w1_plot)), p.w1_plot, "g-", "w1")
    perceptronResult.plotData(range(0, len(p.b_plot)), p.b_plot, "r-", "b")
    perceptronResult.plotData(range(0, len(p.error_plot)), p.error_plot, "c-", "error")
    perceptronResult.limit_y(-2, 2)
    perceptronResult.updateFigure()

    classifier = PlotFigures("Feature space with classifier seperater - The perceptron")
    classifier.plotData(p.wx, p.wy, 'b-', 'The perceptron line')
    classifier.setXlabel('Area')
    classifier.setYlabel('Compactness')
    classifier.plotData(normData.area_norm_training, normData.compactness_norm_training, 'g*', 'Normalized training data')
    classifier.limit_y(0.5, 1)
    classifier.limit_x(0, 1)
    perceptronResult.updateFigure()

    # Do the classification of the normalized testing data and draw on the testing image
    p.doClassification(normData.testing_data_normed, testData.img)

    # Plot the classification
    classifierResult = PlotFigures("Classification of the testing data")
    classifierResult.plotData(p.class1_area, p.class1_compactness, 'ro', 'Class1 classified data')
    classifierResult.plotData(p.classNeg1_area, p.classNeg1_compactness, 'bo', 'ClassNeg1 classified data')
    classifierResult.setXlabel('Area')
    classifierResult.setYlabel('Compactness')
    classifierResult.limit_y(0.5, 1)
    classifierResult.limit_x(0, 1)
    classifierResult.plotData(p.wx, p.wy, 'b-', 'The perceptron line')
    classifierResult.updateFigure()

    # Show the final result of the testing image
    testData.showImage("The classification result of the testing data")

    # Wait here, while user hits ESC.
    while 1:
        # video.getFrame()
        cv2.imshow('Video', video.frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print("User closed the program...")
            break

    # Close down all open windows...
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()