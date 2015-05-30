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
    def __init__(self, listOfFeaturesClass1, listOfFeaturesClass2, listOfFeaturesClass3, featureIndexX, featureIndexY, vizualizeTraining, saveImagePath, normalization):
        self.saveImagePath = saveImagePath
        self.imgClassified = []
        self.maxX = 1
        self.maxY = 1
        self.Xlabel = self.getFeatureLabel(featureIndexX)
        self.Ylabel = self.getFeatureLabel(featureIndexY)

        print "The normalization flag is:", normalization
        print "The vizualize flag is:", vizualizeTraining

        # Load the training data from class1, class2 and class3
        if normalization:
            print "So normalization is true"
            # The length of each list is stored. The X is only saved, since the Y feature list is the same length
            lengthListOfFeaturesClass1X = len(listOfFeaturesClass1[featureIndexX])
            lengthListOfFeaturesClass2X = len(listOfFeaturesClass2[featureIndexX])
            lengthListOfFeaturesClass3X = len(listOfFeaturesClass3[featureIndexX])

            print "lengthListOfFeaturesClass1X", lengthListOfFeaturesClass1X
            print "lengthListOfFeaturesClass2X", lengthListOfFeaturesClass2X
            print "lengthListOfFeaturesClass3X", lengthListOfFeaturesClass3X

            # Add together to one list, in order to normalize all the lists together
            totalListOfFeaturesX = listOfFeaturesClass1[featureIndexX] + listOfFeaturesClass2[featureIndexX] + listOfFeaturesClass3[featureIndexX]
            totalListOfFeaturesY = listOfFeaturesClass1[featureIndexY] + listOfFeaturesClass2[featureIndexY] + listOfFeaturesClass3[featureIndexY]

            # Convert to first numpy array
            np_TotalListOfFeaturesX = np.array(totalListOfFeaturesX)
            np_TotalListOfFeaturesY = np.array(totalListOfFeaturesY)

            # Ops, the normalization process set integers to either 0 or 1. So in order to avoid this, we need to type the
            # input to floats, so eks. 24/129 --> 0.18 instead of 0.
            np_TotalListOfFeaturesY = map(float, np_TotalListOfFeaturesY)
            np_TotalListOfFeaturesY = np.array(np_TotalListOfFeaturesY)

            # Using the CV normalize function
            cv2.normalize(np_TotalListOfFeaturesX, np_TotalListOfFeaturesX, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(np_TotalListOfFeaturesY, np_TotalListOfFeaturesY, 0, 1, cv2.NORM_MINMAX)

            # Now extract the normalized data piece by piece.
            # We now that class1 data must be from 0 to lengthListOfFeaturesClass1X
            # Then class2 data is from index of lengthListOfFeaturesClass1X to the sum between lengthListOfFeaturesClass1X and lengthListOfFeaturesClass2X
            # etc with class 3 X values.
            class1X = np_TotalListOfFeaturesX[0:lengthListOfFeaturesClass1X]
            class2X = np_TotalListOfFeaturesX[lengthListOfFeaturesClass1X:lengthListOfFeaturesClass1X + lengthListOfFeaturesClass2X]
            class3X = np_TotalListOfFeaturesX[lengthListOfFeaturesClass1X + lengthListOfFeaturesClass2X:lengthListOfFeaturesClass1X + lengthListOfFeaturesClass2X + lengthListOfFeaturesClass3X]
            class1Y = np_TotalListOfFeaturesY[0:lengthListOfFeaturesClass1X]
            class2Y = np_TotalListOfFeaturesY[lengthListOfFeaturesClass1X:lengthListOfFeaturesClass1X + lengthListOfFeaturesClass2X]
            class3Y = np_TotalListOfFeaturesY[lengthListOfFeaturesClass1X + lengthListOfFeaturesClass2X:lengthListOfFeaturesClass1X + lengthListOfFeaturesClass2X + lengthListOfFeaturesClass3X]

            # Convert the data into a format that is suitable for the SVM with 3 classes NORMALIZED
            class1Data, class1Label = self.convertDataToSVMFormat3classes(class1X, class1Y, listOfFeaturesClass1[7])
            class2Data, class2Label = self.convertDataToSVMFormat3classes(class2X, class2Y, listOfFeaturesClass2[7])
            class3Data, class3Label = self.convertDataToSVMFormat3classes(class3X, class3Y, listOfFeaturesClass3[7])

        else:
            print "So normalization is false"
            # Convert the data into a format that is suitable for the SVM with 3 classes NOT NORMALIZED
            class1Data, class1Label = self.convertDataToSVMFormat3classes(listOfFeaturesClass1[featureIndexX],
                                                   listOfFeaturesClass1[featureIndexY],
                                                   listOfFeaturesClass1[7])

            class2Data, class2Label = self.convertDataToSVMFormat3classes(listOfFeaturesClass2[featureIndexX],
                                           listOfFeaturesClass2[featureIndexY],
                                           listOfFeaturesClass2[7])

            class3Data, class3Label = self.convertDataToSVMFormat3classes(listOfFeaturesClass3[featureIndexX],
                                           listOfFeaturesClass3[featureIndexY],
                                           listOfFeaturesClass3[7])
        if vizualizeTraining:
            print "So vizualizeTraining is true"
            if normalization:
                print "... and normalization is true"
                featureplot = PlotFigures(1, "Feature plot for training data class 1,2,3 \n",
                                          "with respectively number of samples: " +
                                          str(len(listOfFeaturesClass1[featureIndexX])) + "," +
                                          str(len(listOfFeaturesClass2[featureIndexX])) + "," +
                                          str(len(listOfFeaturesClass3[featureIndexX])), saveImagePath)
                featureplot.plotData(class1X, class1Y, "bs", "class 1")
                featureplot.plotData(class2X, class2Y, "rs", "class 2")
                featureplot.plotData(class3X, class3Y, "ys", "class 3")
                featureplot.setXlabel(self.Xlabel)
                featureplot.setYlabel(self.Ylabel)
                featureplot.limit_x(0, self.maxX)
                featureplot.limit_y(0, self.maxY)
                featureplot.addLegend()
                featureplot.updateFigure()
                featureplot.saveFigure("FeaturePlotForTrainingDataNormalized")
            else:
                print "... but normalization is false"
                featureplot = PlotFigures(1, "Feature plot for training data class 1,2,3 \n",
                                          "with respectively number of samples: " +
                                          str(len(listOfFeaturesClass1[featureIndexX])) + "," +
                                          str(len(listOfFeaturesClass2[featureIndexX])) + "," +
                                          str(len(listOfFeaturesClass3[featureIndexX])), saveImagePath)
                featureplot.plotData(listOfFeaturesClass1[featureIndexX], listOfFeaturesClass1[featureIndexY], "bs", "class 1")
                featureplot.plotData(listOfFeaturesClass2[featureIndexX], listOfFeaturesClass2[featureIndexY], "rs", "class 2")
                featureplot.plotData(listOfFeaturesClass3[featureIndexX], listOfFeaturesClass3[featureIndexY], "ys", "class 3")
                # featureplot.plotData(class1X, class1Y, "bs", "class 1")
                # featureplot.plotData(class2X, class2Y, "rs", "class 2")
                # featureplot.plotData(class3X, class3Y, "ys", "class 3")
                featureplot.setXlabel(self.Xlabel)
                featureplot.setYlabel(self.Ylabel)
                featureplot.addLegend()
                featureplot.updateFigure()
                featureplot.saveFigure("FeaturePlotForTrainingDataNotNormalized")
        # Here we stack the normalized data, i.e. the SVM runs on normalized data
        X, y = self.stackData3classes(class1Data, class2Data, class3Data, class1Label, class2Label, class3Label)

        # SVM regularization parameter for the radial basis function.
        # http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

        # The C parameter trades off misclassification of training examples against simplicity of the decision surface.
        # A low C makes the decision surface smooth, while a high C aims at classifying all
        # training examples correctly by give the model freedom to select more samples as support vectors.
        C = 10

        # The radial base function gamma parameter
        # Intuitively, the gamma parameter defines how far the influence of a single training example
        # reaches, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters
        # can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.
        # If gamma is too large, the radius of the area of influence of the support vectors only includes the support vector
        # it-self and no amount of regularization with C will be able to prevent of overfitting
        gamma = 0.8

        # Step size in the mesh
        self.h = 0.001
        self.xx, self.yy, self.Z, kernel, gamma = self.runSVM(X, y, C, gamma, self.h)

        # Visuzalize with 3 classes
        if vizualizeTraining:
            if normalization:
                # svmPlot = PlotFigures(2, "SVM classification training using a " + kernel + " kernel", "gamma =" + str(gamma) + " and C =" + str(C), saveImagePath)

                if kernel == "rbf":
                    svmPlot = PlotFigures(2, "SVM classification training using a " + kernel + " kernel", "Gamma = " + str(gamma) + " and C = " +str(C), saveImagePath)
                else:
                    svmPlot = PlotFigures(2, "SVM classification training using a " + kernel + " kernel", "", saveImagePath)

                svmPlot.plotContourf(self.xx, self.yy, self.Z)

                # Plot also the training points
                svmPlot.plotData(class1X, class1Y, "bs", "class 1")
                svmPlot.plotData(class2X, class2Y, "rs", "class 2")
                svmPlot.plotData(class3X, class3Y, "ys", "class 3")
                svmPlot.setXlabel(self.Xlabel)
                svmPlot.setYlabel(self.Ylabel)
                svmPlot.limit_x(0, self.maxX)
                svmPlot.limit_y(0, self.maxY)
                # svmPlot.setTitle("SVM classification with training using a linear kernel")
                svmPlot.addLegend()
                svmPlot.updateFigure()
                svmPlot.saveFigure("FeaturePlotForTrainingDataWithBounderyNormalized")

                # print "Stop the program now!!!"
                imgStop = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/imgDrawTest2.png", cv2.CV_LOAD_IMAGE_COLOR)
                cv2.imshow("Stop", imgStop)
                cv2.waitKey(0)

            else:
                svmPlot = PlotFigures(2, "SVM classification training using a " + kernel + " kernel", "gamma =" + str(gamma) + " and C =" + str(C), saveImagePath)
                svmPlot.plotContourf(self.xx, self.yy, self.Z)
                # Plot also the training points
                svmPlot.plotData(listOfFeaturesClass1[featureIndexX], listOfFeaturesClass1[featureIndexY], "bs", "class 1")
                svmPlot.plotData(listOfFeaturesClass2[featureIndexX], listOfFeaturesClass2[featureIndexY], "rs", "class 2")
                svmPlot.plotData(listOfFeaturesClass3[featureIndexX], listOfFeaturesClass3[featureIndexY], "ys", "class 3")
                svmPlot.setXlabel(self.Xlabel)
                svmPlot.setYlabel(self.Ylabel)
                # svmPlot.limit_x(0, self.maxX)
                # svmPlot.limit_y(0, self.maxY)
                # svmPlot.setTitle("SVM classification with training using a linear kernel")
                svmPlot.addLegend()
                svmPlot.updateFigure()
                svmPlot.saveFigure("FeaturePlotForTrainingDataWithBounderyNotNormalized")

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
            return "Width/Length ratio of OBB"
        elif featureIndex == 4:
            return "Number of pixels in OBB"
        elif featureIndex == 5:
            return "Hue mean of pixels in OBB"
        elif featureIndex == 6:
            return "Hue std of pixels in OBB"
        elif featureIndex == 7:
            return "ClassStamp"

    def doClassification(self, testDataX, testDataY):
        Zlist = []
        # print "The length of Z is:", len(self.Z), "and the shape is:", self.Z.shape
        # print "The Z contains this flipped:", np.flipud(self.Z)

        # Then we normalize the data
        normTestDataX = self.NormalizeData(testDataX)
        normTestDataY = self.NormalizeData(testDataY)

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

    def runSVM(self, X, y, C, gamma, h):

        print "Initializing the SVM..."
        degree = 2

        kernel = 'sigmoid'
        kernel = 'poly'
        kernel = 'linear'
        kernel = 'rbf'

        svc = svm.SVC(kernel=kernel, gamma=gamma, degree=degree, C=C).fit(X, y)
        xx, yy = np.meshgrid(np.arange(0, 1, h), np.arange(0, 1, h))

        # Starting the SVM...
        print "Starting the SVM..."

        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        return xx, yy, Z, kernel, gamma

    def convertDataToSVMFormat(self, feature1, feature2, classStamp):
        a = np.array(feature1)
        b = np.array(feature2)
        X = np.column_stack((a,b))
        y = np.array(classStamp)
        return X, y

    def convertDataToSVMFormat3classes(self, featureX, featureY, classStamp):
        a = np.array(featureX)
        b = np.array(featureY)
        data = np.column_stack((a,b))
        label = np.array(classStamp)
        return data, label

    def stackData(self, class1X, classNeg1X, class1y, classNeg1y):
        # Try to stack the X together
        X = np.vstack((class1X,classNeg1X))
        y = np.hstack((class1y,classNeg1y))
        return X, y

    def stackData3classes(self, class1Data, class2Data, class3Data, class1Label, class2Label, class3Label):
        # Try to stack the X together

        # print "So the class1Data before staking is", class1Data
        # print "So the class2Data before staking is", class2Data
        # print "So the class3Data before staking is", class3Data
        #
        # print "So the class1Label before staking is", class1Label
        # print "So the class2Label before staking is", class2Label
        # print "So the class3Label before staking is", class3Label

        X = np.vstack((class1Data,class2Data, class3Data))
        y = np.hstack((class1Label,class2Label, class3Label))

        # print "So X after staking is", X
        # print "So y after staking is", y

        return X, y