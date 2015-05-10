#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
from Segmentation import Segmentation
from PlotFigures import PlotFigures
from sklearn import svm, datasets
import matplotlib.pyplot as plt

class Classification(object):
    # With 3 classes
    # def __init__(self,
    #              featureLengthListClass1,
    #              featureNumberOfSproutPixelsListClass1,
    #              featureClassStampListClass1,
    #
    #              featureLengthListClass2,
    #              featureNumberOfSproutPixelsListClass2,
    #              featureClassStampListClass2,
    #
    #              featureLengthListClass3,
    #              featureNumberOfSproutPixelsListClass3,
    #              featureClassStampListClass3,
    #
    #              vizualizeTraining):

    def __init__(self, listOfFeaturesClass1, listOfFeaturesClass2, listOfFeaturesClass3, featureIndexX, featureIndexY, vizualizeTraining):
        self.imgClassified = []
        self.maxX = 1
        self.maxY = 1

        # self.maxX = max(listOfFeaturesClass1[featureIndexX]) + max(listOfFeaturesClass2[featureIndexX])
        # self.maxY = max(listOfFeaturesClass1[featureIndexY]) + max(listOfFeaturesClass2[featureIndexY])
        # self.maxX = 1000 # DONT UNCOMMENT THIS ONE! MY PC WILL CRASH
        # self.maxY = 1000 # DONT UNCOMMENT THIS ONE! MY PC WILL CRASH

        # print "The choosen features for class1 we are look at is:",listOfFeaturesClass1[featureIndexX]
        # print "The choosen features for class1 we are look at is:",listOfFeaturesClass1[featureIndexY]
        # print "The choosen features for class2 we are look at is:",listOfFeaturesClass2[featureIndexX]
        # print "The choosen features for class2 we are look at is:",listOfFeaturesClass2[featureIndexY]

        # print "The maxX is:", self.maxX
        # print "The maxY is:", self.maxY

        self.Xlabel = self.getFeatureLabel(featureIndexX)
        self.Ylabel = self.getFeatureLabel(featureIndexY)

        # Load the training data from class1, class2 and class3
        if False:
            featureplot = PlotFigures(1, "Feature plot for training data class 1,2,3 \n",
                                      "with respectively number of samples: " +
                                      str(len(listOfFeaturesClass1[featureIndexX])) + "," +
                                      str(len(listOfFeaturesClass2[featureIndexX])) + "," +
                                      str(len(listOfFeaturesClass3[featureIndexX])))
            # featureplot.plotData(listOfFeaturesClass1[featureIndexX], listOfFeaturesClass1[featureIndexY], "bs", "class 1")
            # featureplot.plotData(listOfFeaturesClass2[featureIndexX], listOfFeaturesClass2[featureIndexY], "rs", "class 2")
            # featureplot.plotData(listOfFeaturesClass3[featureIndexX], listOfFeaturesClass3[featureIndexY], "ys", "class 3")
            featureplot.plotData(self.NormalizeData(listOfFeaturesClass1[featureIndexX]), self.NormalizeData(listOfFeaturesClass1[featureIndexY]), "bs", "class 1")
            featureplot.plotData(self.NormalizeData(listOfFeaturesClass2[featureIndexX]), self.NormalizeData(listOfFeaturesClass2[featureIndexY]), "rs", "class 2")
            featureplot.plotData(self.NormalizeData(listOfFeaturesClass3[featureIndexX]), self.NormalizeData(listOfFeaturesClass3[featureIndexY]), "ys", "class 3")
            featureplot.setXlabel(self.Xlabel)
            featureplot.setYlabel(self.Ylabel)
            featureplot.limit_x(0, self.maxX)
            featureplot.limit_y(0, self.maxY)
            featureplot.addLegend()
            featureplot.updateFigure()
            featureplot.saveFigure("FeaturePlotForTrainingData")

        # # Convert the data into a format that is suitable for the SVM with 3 classes NOT NORMALIZED
        # class1X, class1y = self.convertDataToSVMFormat3classes(listOfFeaturesClass1[featureIndexX],
        #                                                listOfFeaturesClass1[featureIndexY],
        #                                                listOfFeaturesClass1[7])
        #
        # class2X, class2y = self.convertDataToSVMFormat3classes(listOfFeaturesClass2[featureIndexX],
        #                                        listOfFeaturesClass2[featureIndexY],
        #                                        listOfFeaturesClass2[7])
        #
        # class3X, class3y = self.convertDataToSVMFormat3classes(listOfFeaturesClass3[featureIndexX],
        #                                        listOfFeaturesClass3[featureIndexY],
        #                                        listOfFeaturesClass3[7])

        # Convert the data into a format that is suitable for the SVM with 3 classes NORMALIZED
        class1X, class1y = self.convertDataToSVMFormat3classes(self.NormalizeData(listOfFeaturesClass1[featureIndexX]),
                                                       self.NormalizeData(listOfFeaturesClass1[featureIndexY]),
                                                       listOfFeaturesClass1[7])

        class2X, class2y = self.convertDataToSVMFormat3classes(self.NormalizeData(listOfFeaturesClass2[featureIndexX]),
                                               self.NormalizeData(listOfFeaturesClass2[featureIndexY]),
                                               listOfFeaturesClass2[7])

        class3X, class3y = self.convertDataToSVMFormat3classes(self.NormalizeData(listOfFeaturesClass3[featureIndexX]),
                                               self.NormalizeData(listOfFeaturesClass3[featureIndexY]),
                                               listOfFeaturesClass3[7])

        # Here we stack the normalized data, i.e. the SVM runs on normalized data
        X, y = self.stackData3classes(class1X, class2X, class3X, class1y, class2y, class3y)

        # print "The X after iss has been stacked:", X
        # print "The y after is has been stacked:", y

        # SVM regularization parameter
        C = 1
        # Step size in the mesh
        # self.h = 0.1
        self.h = 0.001
        self.xx, self.yy, self.Z, kernel, gamma = self.runSVM(X, y, C, self.h)
        print "The calculated Z matrix has size:", self.Z.shape
        print "And the Z matrix is:", self.Z

        # print "The xx is:", self.xx
        # print "The yy is:", self.yy
        # print "The Z is:", self.Z

        # Use the classifier to check if an input
        # x = 20
        # y = 50
        # testData = (x, y)
        #
        # print "With a testData point of", testData, "the class is:", self.doClassification(x, y)
        # Plot the testData point
        # svmPlot.plotData(testData[0], testData[1], "gs", "testdata")

        # Visuzalize with 3 classes
        if False:
            svmPlot = PlotFigures(2, "SVM classification training using a " + kernel + " kernel", "gamma =" + str(gamma) + " and C =" + str(C))

            svmPlot.plotContourf(self.xx, self.yy, self.Z)
            # Plot also the training points
            svmPlot.plotData(self.NormalizeData(listOfFeaturesClass1[featureIndexX]), self.NormalizeData(listOfFeaturesClass1[featureIndexY]), "bs", "class 1")
            svmPlot.plotData(self.NormalizeData(listOfFeaturesClass2[featureIndexX]), self.NormalizeData(listOfFeaturesClass2[featureIndexY]), "rs", "class 2")
            svmPlot.plotData(self.NormalizeData(listOfFeaturesClass3[featureIndexX]), self.NormalizeData(listOfFeaturesClass3[featureIndexY]), "ys", "class 3")
            svmPlot.setXlabel(self.Xlabel)
            svmPlot.setYlabel(self.Ylabel)
            svmPlot.limit_x(0, self.maxX)
            svmPlot.limit_y(0, self.maxY)
            # svmPlot.setTitle("SVM classification with training using a linear kernel")
            svmPlot.addLegend()
            svmPlot.updateFigure()
            svmPlot.saveFigure("FeaturePlotForTrainingDataWithBoundery")
        print "Finish with the supervised learning...\n"

    def NormalizeData(self, featureList):
        maxValue = max(featureList)
        normList = np.array(featureList, dtype=np.float64)/maxValue
        return normList

    def getFeatureLabel(self, featureIndex):
        if featureIndex == 0:
            return "Center of mass"
        elif featureIndex == 1:
            return "Length of OBB"
        elif featureIndex == 2:
            return "Width of OBB"
        elif featureIndex == 3:
            return "Ratio of OBB"
        elif featureIndex == 4:
            return "Number of pixels in OBB"
        elif featureIndex == 5:
            return "Hue mean of pixels in OBB"
        elif featureIndex == 6:
            return "Hue std of pixels in OBB"
        elif featureIndex == 7:
            return "ClassStamp"

    # def convertTestDataToZTable(self, testDataX, testDataY):
    #     # First normalize the testDataX and testDataY
    #     normDataX = self.NormalizeData(testDataX)
    #     normDataY = self.NormalizeData(testDataY)
    #
    #     # Get the number of degits, we need to round with, bases on how the stepsize h is
    #     numberOfDegits = int(np.log10(int(1/self.h)))
    #
    #     # Then we round the data, compair with the stepsize h, to be sure that we hit a valid spot in the meshgrid
    #     # I.e. if the grid goes xx = 0.01, 0.02, 0.03 ... 1.0, then a value of 0.013 should get to 0.01. This depends on the h stepsize
    #     roundListX = [round(elem, numberOfDegits) for elem in normDataX]
    #     roundListY = [round(elem, numberOfDegits) for elem in normDataY]
    #
    #     # Then multiply each element with the inverse of the stepsize, e.g. h = 0.1, we multiply with 10
    #     npArrayX = np.array(roundListX)*int((1/self.h))
    #     npArrayY = np.array(roundListY)*int((1/self.h))
    #
    #     # Then we subtract -1 for each element, since our grid goes from 0 - 9 in a (10,10) shape
    #     npArrayXsub = np.array(npArrayX)-1
    #     npArrayYsub = np.array(npArrayY)-1
    #
    #     # Elements which are zeroes will be -1, so
    #     # where elements are -1 we set them to 0.
    #     npArrayXsubNoNeg = np.array(npArrayXsub).clip(min=0)
    #     npArrayYsubNoNeg = np.array(npArrayYsub).clip(min=0)
    #
    #     return npArrayXsubNoNeg, npArrayYsubNoNeg

    def doClassification(self, testDataX, testDataY):
        Zlist = []
        # print "The length of Z is:", len(self.Z), "and the shape is:", self.Z.shape
        # print "The Z contains this flipped:", np.flipud(self.Z)

        print "The input testDataX to the doClassification is:", testDataX, "\n"
        print "The input testDataY to the doClassification is:", testDataY, "\n"

        # Then we normalize the data
        normTestDataX = self.NormalizeData(testDataX)
        normTestDataY = self.NormalizeData(testDataY)

        print "The input testDataX is normalized to this:", normTestDataX, "\n"
        print "The input testDataY is normalized to this:", normTestDataY, "\n"

        # Now we have a normalize dataX and dataY. This coordinate is used as
        # a lookup in the Z-matrix. With h=0.001, we have a Z-matrix shape of (1000, 1000)
        # So with a datapoint of dataX[0] and dataY[0] = (0.14563624 , 0.03587444)
        # this datapoint must be converted to be (after rounding) to be (146 ,36)
        # And in this location we see how the Z-matrix contains of value.

        for element in zip(normTestDataX, normTestDataY):
            # The normalized data X andY is adjusted regarding the stepsize h and rounded
            elementX = int(round(element[0] * 1/self.h, 0))
            elementY = int(round(element[1] * 1/self.h, 0))

            # Instead of swopping x and y, we just look up in a y,x fashion
            # temp = self.Z[element[1]/self.h, element[0]/self.h]
            # In case there the element X and Y is zero,
            # the index of Z-matrix is -1.
            # To compensate for this cheap, we just make a check. Are we having a
            # a (0.0) sample, then we make it to a (1,1) sample.
            # Then we do temp = ..... Z(1-1,1-1) which is Z(0.0)
            if elementX == 0:
                elementX = 1
            if elementY == 0:
                elementY = 1

            temp = self.Z[elementY-1, elementX-1]
            Zlist.append(temp)
        return Zlist

    def getClassifiedLists3classes(self, testDataX, testDataY, centerList, imgRGB):
        self.imgClassified = imgRGB.copy()
        featureClass1ListX = []
        featureClass1ListY = []
        featureClass2ListX = []
        featureClass2ListY = []
        featureClass3ListX = []
        featureClass3ListY = []

        centerClass1List = []
        centerClass2List = []
        centerClass3List = []

        # OK input of testDataX, testDataY and centerList. The imgRGB is tested and it works fine with cv2.imshow(...)
        # The testDataX and testDataY is not normalized now...
        Znew = self.doClassification(testDataX, testDataY)

        for index in zip(Znew, self.NormalizeData(testDataX), self.NormalizeData(testDataY), centerList):
            # print "So the index is", index[0]
            # print "So the x,y is:", index[1], index[2]
            # print "So the center is", index[3]

            # If the Z value at this index is zero
            # print "The index[0], i.e. the Z value is:", index[0]
            if index[0] == 1:       # If class 1, then the color indication is blue
                featureClass1ListX.append(index[1])
                featureClass1ListY.append(index[2])
                centerClass1List.append(index[3])
                cv2.circle(self.imgClassified, index[3], 5, (255, 0, 0), -1)
            elif index[0] == 2:     # If class 2, then the color indication is red
                featureClass2ListX.append(index[1])
                featureClass2ListY.append(index[2])
                centerClass2List.append(index[3])
                cv2.circle(self.imgClassified, index[3], 5, (0, 0, 255), -1)
            elif index[0] == 3:     #if the class is class 3, and the color indication is yellow
                featureClass3ListX.append(index[1])
                featureClass3ListY.append(index[2])
                centerClass3List.append(index[3])
                cv2.circle(self.imgClassified, index[3], 5, (0, 255, 255), -1)
            else:
                print "What are we doing here?"

        # Returning with 3 classes
        return featureClass1ListX, \
               featureClass1ListY, \
               centerClass1List, \
               featureClass2ListX, \
               featureClass2ListY, \
               centerClass2List, \
               featureClass3ListX, \
               featureClass3ListY, \
               centerClass3List

    def runSVM(self, X, y, C, h):

        print "Initializing the SVM..."
        kernel = 'rbf'
        # Intial gamma was 0.7
        gamma = 0.8
        # kernel = 'linear'
        # kernel = 'poly'

        # svc = svm.SVC(kernel=kernel, C=C).fit(X, y)
        svc = svm.SVC(kernel=kernel, gamma=gamma, C=C).fit(X, y)
        # svc = svm.SVC(kernel=kernel, degree=2, C=C).fit(X, y)
        # svc = svm.LinearSVC(C=C).fit(X, y)

        # Starting the SVM...
        print "Starting the SVM..."

        # Create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        print "x_min is:", x_min
        print "x_max is:", x_max
        print "y_min", y_min
        print "y_max", y_max

        xx, yy = np.meshgrid(np.arange(0, 1, h), np.arange(0, 1, h))

        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # xx, yy = np.meshgrid(np.arange(0, self.maxX, h), np.arange(0, self.maxY, h))

        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        print "The Z computed directly from scv is", Z

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        return xx, yy, Z, kernel, gamma

    # def plotMesh(self, X):
    #     # Create a mesh to plot in
    #     # Step size in the mesh
    #     h = .02
    #     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #     return xx, yy

    def convertDataToSVMFormat(self, feature1, feature2, classStamp):
        a = np.array(feature1)
        b = np.array(feature2)
        X = np.column_stack((a,b))
        y = np.array(classStamp)
        return X, y

    def convertDataToSVMFormat3classes(self, feature1, feature2, classStamp):
        a = np.array(feature1)
        b = np.array(feature2)
        X = np.column_stack((a,b))
        y = np.array(classStamp)
        return X, y

    def stackData(self, class1X, classNeg1X, class1y, classNeg1y):
        # Try to stack the X together
        X = np.vstack((class1X,classNeg1X))
        y = np.hstack((class1y,classNeg1y))
        return X, y

    def stackData3classes(self, class1X, class2X, class3X, class1y, class2y, class3y):
        # Try to stack the X together
        X = np.vstack((class1X,class2X, class3X))
        y = np.hstack((class1y,class2y, class3y))
        return X, y