#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
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
        featureplot = PlotFigures("Feature space for training data class 1 and class -1", "FeatureSpaceClass1andClassNeg1")
        featureplot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
        featureplot.plotData(featureLengthListClassNeg1, featureNumberOfSproutPixelsListClassNeg1, "bs", "class -1")
        featureplot.setXlabel("Length of sprout bounding box")
        featureplot.setYlabel("Number of sprout pixels in bounding box")
        featureplot.limit_x(0,self.maxX)
        featureplot.limit_y(0,self.maxY)
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
        h = 0.1
        xx, yy, Z = self.runSVM(X, y, C, h)

        # print "The shape of Z is:", Z.shape
        # print "This is Z", Z
        # print "And now the flipUD Z is:", np.flipud(Z)

        svmPlot = PlotFigures("SVM classification with training using a linear kernel", "SVMlinearKernel")
        svmPlot.plotContourf(xx, yy, Z)

        # print "So xx is:", xx
        # print "So yy is:", yy
        # print "The length of xx is:", len(xx)
        # print "The length of yy is:", len(yy)
        # print "The length of Z is:", len(Z)

        # Use the classifier to check if an input
        testData = (20, 35)
        print "With a testData point of", testData, "the class is:", self.doClassification(Z, testData, h)

        # Plot the testData point
        svmPlot.plotData(testData[0], testData[1], "gs", "testdata")

        # Plot also the training points
        svmPlot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
        svmPlot.plotData(featureLengthListClassNeg1, featureNumberOfSproutPixelsListClassNeg1, "bs", "class -1")
        svmPlot.setXlabel("Length of sprout bounding box")
        svmPlot.setYlabel("Number of sprout pixels in bounding box")
        svmPlot.limit_x(0, self.maxX)
        svmPlot.limit_y(0, self.maxY)
        svmPlot.updateFigure()

        print "Are we there yet?"

    def doClassification(self, Z, testData, h):
        # Instead of swopping x and y, we just look up in a y,x fashion
        return Z[testData[1]/h, testData[0]/h]

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

    def functionFromClassification(self):
        print "Now the functionFromClassification is called"