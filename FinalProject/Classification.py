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

        # Load the training data from class1 and class -1
        pf = PlotFigures("Feature space for training data class 1 and class -1", "FeatureSpaceClass1andClassNeg1")
        pf.plotData(featureLengthListClass1, featureNumberOfSproutPixelsListClass1, "rs", "class 1")
        pf.plotData(featureLengthListClassNeg1, featureNumberOfSproutPixelsListClassNeg1, "bs", "class -1")
        pf.setXlabel("Length of sprout bounding box")
        pf.setYlabel("Number of sprout pixels in bounding box")
        # pf.updateFigure()

        class1X, class1y = self.convertDataToSVMFormat(featureLengthListClass1,
                                                       featureNumberOfSproutPixelsListClass1,
                                                       featureClassStampListClass1)
        classNeg1X, classNeg1y = self.convertDataToSVMFormat(featureLengthListClassNeg1,
                                               featureNumberOfSproutPixelsListClassNeg1,
                                               featureClassStampListClassNeg1)

        X,y = self.stackData(class1X, classNeg1X, class1y, classNeg1y)

        # And the X and Y crooped is:
        # print "X cropped to:", X[0:2]

        C = 1.0
        svc = svm.SVC(kernel='linear', C=C).fit(X, y)
        # rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        # poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
        # lin_svc = svm.LinearSVC(C=C).fit(X, y)

        # Create a mesh to plot in
        # Step size in the mesh
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # # Title for the plots
        titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

        plt.subplot(1, 1, 1)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        # Starting the SVM...
        print "Starting the SVM..."
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.title(titles[0])
        plt.show()

        # for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        #     # Plot the decision boundary. For that, we will assign a color to each
        #     # point in the mesh [x_min, m_max]x[y_min, y_max].
        #     plt.subplot(2, 2, i + 1)
        #     plt.subplots_adjust(wspace=0.4, hspace=0.4)
        #
        #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        #
        #     # Put the result into a color plot
        #     Z = Z.reshape(xx.shape)
        #     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        #
        #     # Plot also the training points
        #     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        #     plt.xlabel('Sepal length')
        #     plt.ylabel('Sepal width')
        #     plt.xlim(xx.min(), xx.max())
        #     plt.ylim(yy.min(), yy.max())
        #     plt.xticks(())
        #     plt.yticks(())
        #     plt.title(titles[i])
        # plt.show()

        print "Are we there yet?"

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