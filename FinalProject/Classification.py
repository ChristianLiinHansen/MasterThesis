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
    def __init__(self,
                 featureLengthListClass1,
                 featureNumberOfSproutPixelsListClass1,
                 featureClassStampListClass1,
                 featureLengthListClassNeg1,
                 featureNumberOfSproutPixelsListClassNeg1,
                 featureClassStampListClassNeg1):

        self.maxX = 50
        self.maxY = 255

        # Load the training data from class1 and class -1
        featureplot = PlotFigures(1)
        featureplot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
        featureplot.plotData(featureLengthListClassNeg1, featureNumberOfSproutPixelsListClassNeg1, "bs", "class -1")

        self.Xlabel = "Length of sprout bounding box"
        self.Ylabel = "Number of sprout pixels in bounding box"

        featureplot.setXlabel(self.Xlabel)
        featureplot.setYlabel(self.Ylabel)
        featureplot.limit_x(0, self.maxX)
        featureplot.limit_y(0, self.maxY)
        featureplot.setTitle("Feature space for training data class 1 and class -1")
        featureplot.addLegend()
        featureplot.updateFigure()

        # Convert the data into a format that is suitable for the SVM
        class1X, class1y = self.convertDataToSVMFormat(featureLengthListClass1,
                                                       featureNumberOfSproutPixelsListClass1,
                                                       featureClassStampListClass1)
        classNeg1X, classNeg1y = self.convertDataToSVMFormat(featureLengthListClassNeg1,
                                               featureNumberOfSproutPixelsListClassNeg1,
                                               featureClassStampListClassNeg1)

        X, y = self.stackData(class1X, classNeg1X, class1y, classNeg1y)

        # SVM regularization parameter
        C = 1.0
        # Step size in the mesh
        self.h = 0.1
        self.xx, self.yy, self.Z = self.runSVM(X, y, C, self.h)

        svmPlot = PlotFigures(2)
        svmPlot.plotContourf(self.xx, self.yy, self.Z)

        # Use the classifier to check if an input
        # x = 20
        # y = 50
        # testData = (x, y)
        #
        # print "With a testData point of", testData, "the class is:", self.doClassification(x, y)

        # Plot the testData point
        # svmPlot.plotData(testData[0], testData[1], "gs", "testdata")

        # Plot also the training points
        svmPlot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
        svmPlot.plotData(featureLengthListClassNeg1, featureNumberOfSproutPixelsListClassNeg1, "bs", "class -1")
        svmPlot.setXlabel("Length of sprout bounding box")
        svmPlot.setYlabel("Number of sprout pixels in bounding box")
        svmPlot.limit_x(0, self.maxX)
        svmPlot.limit_y(0, self.maxY)
        svmPlot.setTitle("SVM classification with training using a linear kernel")
        svmPlot.addLegend()
        svmPlot.updateFigure()

        print "Finish with the supervised learning..."

    def doClassification(self, testDataX, testDataY):
        Zlist = []
        for element in zip(testDataX, testDataY):
            # Instead of swopping x and y, we just look up in a y,x fashion
            temp = self.Z[element[1]/self.h, element[0]/self.h]
            Zlist.append(temp)
        return Zlist

    def getClassifiedLists(self, testDataX, testDataY, centerList, imgRGB):
        imgClassify = imgRGB.copy()
        featureClass1ListX = []
        featureClass1ListY = []
        featureClassNeg1ListX = []
        featureClassNeg1ListY = []

        centerClass1List = []
        centerClassNeg1List = []

        # OK input of testDataX, testDataY and centerList. The imgRGB is tested and it works fine with cv2.imshow(...)
        # print "Inside getClassifiedList, the testDataX is:", testDataX, "and the length is:", len(testDataX)
        # print "Inside getClassifiedList, the testDataY is:", testDataY, "and the length is:", len(testDataY)
        # print "Inside getClassifiedList, the centerList is:", centerList, "and the length is:", len(centerList)

        Znew = self.doClassification(testDataX, testDataY)
        # print "The Znew is:", Znew, "with a length of:", len(Znew)

        for index in zip(Znew, testDataX, testDataY, centerList):
            # print "So the index is", index[0]
            # print "So the x,y is:", index[1], index[2]
            # print "So the center is", index[3]

            # If the Z value at this index is zero
            if index[0] == 1:
                featureClass1ListX.append(index[1])
                featureClass1ListY.append(index[2])
                centerClass1List.append(index[3])
                cv2.circle(imgClassify, index[3], 5, (0, 0, 255), -1)
            else:
                featureClassNeg1ListX.append(index[1])
                featureClassNeg1ListY.append(index[2])
                centerClassNeg1List.append(index[3])
                cv2.circle(imgClassify, index[3], 5, (255, 0, 0), -1)
        # Show the classified result
        cv2.imshow("Classified result", imgClassify)
        return featureClass1ListX, featureClass1ListY, featureClassNeg1ListX, featureClassNeg1ListY, centerClass1List, centerClassNeg1List

    def runSVM(self, X, y, C, h):

        print "Initializing the SVM..."
        svc = svm.SVC(kernel='linear', C=C).fit(X, y)
        # svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        # svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)
        # svc = svm.LinearSVC(C=C).fit(X, y)

        # Starting the SVM...
        print "Starting the SVM..."

        # Create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        xx, yy = np.meshgrid(np.arange(0, self.maxX, h), np.arange(0, self.maxY, h))

        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        return xx, yy, Z

    def plotMesh(self, X):
        # Create a mesh to plot in
        # Step size in the mesh
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def convertDataToSVMFormat(self, feature1, feature2, classStamp):
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