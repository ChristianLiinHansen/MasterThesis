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
    # With 2 working classes
    # def __init__(self,
    #              featureLengthListClass1,
    #              featureNumberOfSproutPixelsListClass1,
    #              featureClassStampListClass1,
    #
    #              featureLengthListClassNeg1,
    #              featureNumberOfSproutPixelsListClassNeg1,
    #              featureClassStampListClassNeg1,
    #
    #              vizualizeTraining):

    # With 3 classes
    def __init__(self,
                 featureLengthListClass1,
                 featureNumberOfSproutPixelsListClass1,
                 featureClassStampListClass1,

                 featureLengthListClass2,
                 featureNumberOfSproutPixelsListClass2,
                 featureClassStampListClass2,

                 featureLengthListClass3,
                 featureNumberOfSproutPixelsListClass3,
                 featureClassStampListClass3,

                 vizualizeTraining):

        self.maxX = 100
        self.maxY = 255

        self.Xlabel = "Length of sprout bounding box"
        self.Ylabel = "Number of sprout pixels in bounding box"

        # # Load the training data from class1 and class -1
        # if vizualizeTraining:
        #     featureplot = PlotFigures(1)
        #     featureplot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
        #     featureplot.plotData(featureLengthListClassNeg1, featureNumberOfSproutPixelsListClassNeg1, "bs", "class -1")
        #     featureplot.setXlabel(self.Xlabel)
        #     featureplot.setYlabel(self.Ylabel)
        #     featureplot.limit_x(0, self.maxX)
        #     featureplot.limit_y(0, self.maxY)
        #     featureplot.setTitle("Feature space for training data class 1 and class -1")
        #     featureplot.addLegend()
        #     featureplot.updateFigure()

        # Load the training data from class1, class2 and class3
        if vizualizeTraining:
            featureplot = PlotFigures(1)
            featureplot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
            featureplot.plotData(featureLengthListClass2, featureNumberOfSproutPixelsListClass2, "bs", "class 2")
            featureplot.plotData(featureLengthListClass3, featureNumberOfSproutPixelsListClass3, "gs", "class 3")
            featureplot.setXlabel(self.Xlabel)
            featureplot.setYlabel(self.Ylabel)
            featureplot.limit_x(0, self.maxX)
            featureplot.limit_y(0, self.maxY)
            featureplot.setTitle("Feature space for training data class 1, 2 and 3")
            featureplot.addLegend()
            featureplot.updateFigure()


        imgBreak = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        cv2.imshow("Show what we have is...", imgBreak)
        cv2.waitKey(0)

        # Convert the data into a format that is suitable for the SVM with 2 classes
        # class1X, class1y = self.convertDataToSVMFormat(featureLengthListClass1,
        #                                                featureNumberOfSproutPixelsListClass1,
        #                                                featureClassStampListClass1)
        # classNeg1X, classNeg1y = self.convertDataToSVMFormat(featureLengthListClassNeg1,
        #                                        featureNumberOfSproutPixelsListClassNeg1,
        #                                        featureClassStampListClassNeg1)
        #
        # X, y = self.stackData(class1X, classNeg1X, class1y, classNeg1y)

        # Convert the data into a format that is suitable for the SVM with 3 classes
        class1X, class1y = self.convertDataToSVMFormat3classes(featureLengthListClass1,
                                                       featureNumberOfSproutPixelsListClass1,
                                                       featureClassStampListClass1)

        class2X, class2y = self.convertDataToSVMFormat3classes(featureLengthListClass2,
                                               featureNumberOfSproutPixelsListClass2,
                                               featureClassStampListClass2)

        class3X, class3y = self.convertDataToSVMFormat3classes(featureLengthListClass3,
                                               featureNumberOfSproutPixelsListClass3,
                                               featureClassStampListClass3)


        X, y = self.stackData3classes(class1X, class2X, class3X, class1y, class2y, class3y)

        # SVM regularization parameter
        C = 1.0
        # Step size in the mesh
        # self.h = 0.1
        self.h = 0.1
        self.xx, self.yy, self.Z = self.runSVM(X, y, C, self.h)

        # Use the classifier to check if an input
        # x = 20
        # y = 50
        # testData = (x, y)
        #
        # print "With a testData point of", testData, "the class is:", self.doClassification(x, y)
        # Plot the testData point
        # svmPlot.plotData(testData[0], testData[1], "gs", "testdata")

        # # Visuzalize with 2 classes
        # if vizualizeTraining:
        #     svmPlot = PlotFigures(2)
        #     svmPlot.plotContourf(self.xx, self.yy, self.Z)
        #     # Plot also the training points
        #     svmPlot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
        #     svmPlot.plotData(featureLengthListClassNeg1, featureNumberOfSproutPixelsListClassNeg1, "bs", "class -1")
        #     svmPlot.setXlabel("Length of sprout bounding box")
        #     svmPlot.setYlabel("Number of sprout pixels in bounding box")
        #     svmPlot.limit_x(0, self.maxX)
        #     svmPlot.limit_y(0, self.maxY)
        #     svmPlot.setTitle("SVM classification with training using a linear kernel")
        #     svmPlot.addLegend()
        #     svmPlot.updateFigure()

        # Visuzalize with 3 classes
        if vizualizeTraining:
            svmPlot = PlotFigures(2)
            svmPlot.plotContourf(self.xx, self.yy, self.Z)
            # Plot also the training points
            svmPlot.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
            svmPlot.plotData(featureLengthListClass2, featureNumberOfSproutPixelsListClass2, "bs", "class 2")
            svmPlot.plotData(featureLengthListClass3, featureNumberOfSproutPixelsListClass3, "gs", "class 3")
            svmPlot.setXlabel("Length of sprout bounding box")
            svmPlot.setYlabel("Number of sprout pixels in bounding box")
            svmPlot.limit_x(0, self.maxX)
            svmPlot.limit_y(0, self.maxY)
            svmPlot.setTitle("SVM classification with training using a linear kernel")
            svmPlot.addLegend()
            svmPlot.updateFigure()

        print "Finish with the supervised learning...\n"

    def doClassification(self, testDataX, testDataY):
        Zlist = []
        # print "The length of Z is:", len(self.Z), "and the shape is:", self.Z.shape
        # With h = 0.1, the shape of Z is: (2550, 500), i.e. 2550 cols and 500 rows.
        for element in zip(testDataX, testDataY):
            # Instead of swopping x and y, we just look up in a y,x fashion
            temp = self.Z[element[1]/self.h, element[0]/self.h]
            Zlist.append(temp)
        return Zlist

    # def getClassifiedLists(self, testDataX, testDataY, centerList, imgRGB):
    #     imgClassify = imgRGB.copy()
    #     featureClass1ListX = []
    #     featureClass1ListY = []
    #     featureClassNeg1ListX = []
    #     featureClassNeg1ListY = []
    #
    #     centerClass1List = []
    #     centerClassNeg1List = []
    #
    #     # print "The input to the getClassifier is:", testDataX
    #     # print "The input to the getClassifier is:", testDataY
    #     # print "The input to the getClassifier is:", centerList
    #     # print "The input to the getClassifier is:", imgRGB
    #
    #     # OK input of testDataX, testDataY and centerList. The imgRGB is tested and it works fine with cv2.imshow(...)
    #     # print "Inside getClassifiedList, the testDataX is:", testDataX, "and the length is:", len(testDataX)
    #     # print "Inside getClassifiedList, the testDataY is:", testDataY, "and the length is:", len(testDataY)
    #     # print "Inside getClassifiedList, the centerList is:", centerList, "and the length is:", len(centerList)
    #
    #     Znew = self.doClassification(testDataX, testDataY)
    #     # print "The Znew is:", Znew, "with a length of:", len(Znew)
    #
    #     for index in zip(Znew, testDataX, testDataY, centerList):
    #         # print "So the index is", index[0]
    #         # print "So the x,y is:", index[1], index[2]
    #         # print "So the center is", index[3]
    #
    #         # If the Z value at this index is zero
    #         if index[0] == 1:
    #             featureClass1ListX.append(index[1])
    #             featureClass1ListY.append(index[2])
    #             centerClass1List.append(index[3])
    #             cv2.circle(imgClassify, index[3], 5, (0, 0, 255), -1)
    #         else:
    #             featureClassNeg1ListX.append(index[1])
    #             featureClassNeg1ListY.append(index[2])
    #             centerClassNeg1List.append(index[3])
    #             cv2.circle(imgClassify, index[3], 5, (255, 0, 0), -1)
    #
    #     # Show the classified result
    #     cv2.imshow("Classified result", imgClassify)
    #     return featureClass1ListX, featureClass1ListY, featureClassNeg1ListX, featureClassNeg1ListY, centerClass1List, centerClassNeg1List

    def getClassifiedLists3classes(self, testDataX, testDataY, centerList, imgRGB):
        imgClassify = imgRGB.copy()
        featureClass1ListX = []
        featureClass1ListY = []
        featureClass2ListX = []
        featureClass2ListY = []
        featureClass3ListX = []
        featureClass3ListY = []

        centerClass1List = []
        centerClass2List = []
        centerClass3List = []

        # print "The input to the getClassifier is:", testDataX
        # print "The input to the getClassifier is:", testDataY
        # print "The input to the getClassifier is:", centerList
        # print "The input to the getClassifier is:", imgRGB

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
            elif index[0] == 2:
                featureClass2ListX.append(index[1])
                featureClass2ListY.append(index[2])
                centerClass2List.append(index[3])
                cv2.circle(imgClassify, index[3], 5, (0, 255, 0), -1)
            else: #Otherwise the class is class 3
                featureClass3ListX.append(index[1])
                featureClass3ListY.append(index[2])
                centerClass3List.append(index[3])
                cv2.circle(imgClassify, index[3], 5, (255, 0, 0), -1)

        # Show the classified result
        cv2.imshow("Classified result", imgClassify)
        # Returning with 2 classes
        # return featureClass1ListX, featureClass1ListY, featureClassNeg1ListX, featureClassNeg1ListY, centerClass1List, centerClassNeg1List
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