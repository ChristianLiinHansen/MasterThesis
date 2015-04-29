#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import classes from component files
import cv2
import time
from Input import Input
from Preprocessing import Preprocessing
from Segmentation import Segmentation
from Classification import Classification
from Output import Output
from PlotFigures import PlotFigures
from pylab import *


def main():

    ######################################################################
    # In the begging, the whole system starts learning, using provided training data.
    # I.e. all the training images is handled within the constructor of each class.
    # Each class should have a function of do<NameOfClass>, e.g doSegmentation() with the argument from the documentation diagram
    # page 12, "Chain of the project"
    ######################################################################

    # Initialize the Input component with cameraIndex = 0 (webcamera inbuilt in PC)
    # Input: Plug and play webcamera
    # Output: RGB image, training data and testing data
    i = Input(0)

    # Initialize the Preprocessing component with the training data1 and -1
    # Input: trainingData1, trainingDataNeg1
    # Output: imgThreshold, imgSeedAndSprout.
    p1 = Preprocessing(i.trainingData1, 1)
    pNeg1 = Preprocessing(i.trainingDataNeg1, -1)

    # Initializing the Segmentation component
    s1 = Segmentation(i.trainingData1, p1.imgFrontGround, p1.imgSeedAndSprout, p1.imgSprout, 1)
    sNeg1 = Segmentation(i.trainingDataNeg1, pNeg1.imgFrontGround, pNeg1.imgSeedAndSprout, pNeg1.imgSprout, -1)

    # Initialize the Classification component
    c = Classification(s1.featureLengthList,
                       s1.featureNumberOfSproutPixelsList,
                       s1.featureClassStampList,
                       sNeg1.featureLengthList,
                       sNeg1.featureNumberOfSproutPixelsList,
                       sNeg1.featureClassStampList)

    # Initialize the Output component
    o = Output()

    # At this point, the whole system has been taught with supervised learning.
    # Training data has been loaded, preprocessed, segmented, feature extracted and classified.
    # From here, the testing data is loaded by using the webcam, where each seed will be preprocessed, segmented and classified
    # based on what how the line of seperation lies.

    while i.cameraIsOpen:

        # Input from webcamera - Testing data
        # imgInput = i.getImg()
        # cv2.imshow("Streaming from camera", imgInput)

        # # As a beginning, the testing data is for now, just a still image, with a mix of diffrent seeds
        # # Later the imgInput should come from the camera as written above.
        imgInput = i.testingData

        # The input image is processed through each component as followed, with class 0, since it is unknow which class the
        # test image belogns to...
        p = Preprocessing(imgInput, 0)

        # The FrontGround image and SeedAndSprout image is used in the segmentation component
        s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedAndSprout, p.imgSprout, 0)
        cv2.imshow("Show the RGB image with contours of sprouts", s.imgDraw)

        # Plot the featureplot for the testing data, e.i class 0
        featureplotClass0 = PlotFigures(3)
        featureplotClass0.clearFigure()
        featureplotClass0.plotData(s.featureLengthList, s.featureNumberOfSproutPixelsList, "gs", "class 0")
        featureplotClass0.limit_x(0, c.maxX)
        featureplotClass0.limit_y(0, c.maxY)
        featureplotClass0.setTitle("Feature plot for testing data class 0")
        featureplotClass0.addLegend()
        featureplotClass0.setXlabel(c.Xlabel)
        featureplotClass0.setYlabel(c.Ylabel)
        featureplotClass0.updateFigure()

        # Now with the featureplot of class0, we need to draw the featureplot where the class0 is going to get classified.
        # I.e. we want an plot, where the same testing data is seperated into red or blue area, like the training data.
        featureplotClass0Classified = PlotFigures(4)
        featureplotClass0Classified.clearFigure()

        # print "What do we have?..."
        # print "The x feature list is:", s.featureLengthList
        # print "The y feature list is:", s.featureNumberOfSproutPixelsList
        # We combine the x,y feature list into a single list with (x,y) points.
        featureClass1ListX, \
        featureClass1ListY, \
        featureClassNeg1ListX, \
        featureClassNeg1ListY, \
        centerClass1List, \
        centerClassNeg1List \
            = c.getClassifiedLists(s.featureLengthList, s.featureNumberOfSproutPixelsList, s.featureCenterOfMassList, s.imgRGB)

        # Here we plot the data that has been classified...
        featureplotClass0Classified.plotData(featureClass1ListX, featureClass1ListY, "rs", "class 1")
        featureplotClass0Classified.plotData(featureClassNeg1ListX, featureClassNeg1ListY, "bs", "class -1")
        featureplotClass0Classified.plotContourf(c.xx, c.yy, c.Z)
        featureplotClass0Classified.limit_x(0, c.maxX)
        featureplotClass0Classified.limit_y(0, c.maxY)
        featureplotClass0Classified.setTitle("Feature plot for classified test data")
        featureplotClass0Classified.addLegend()
        featureplotClass0Classified.setXlabel(c.Xlabel)
        featureplotClass0Classified.setYlabel(c.Ylabel)
        featureplotClass0Classified.updateFigure()

        # With the list of COM for good and bad seeds, the last component is used
        # Remember that the output now is in cm. Change the z value to 0.30 to get the x,y, in meters,
        # which is needed for the UR-robot. 
        xyzList0, xyzList1 = o.convertUV2XYZ(centerClass1List, centerClassNeg1List, imgInput.shape)
        print "The xyzList0 is:", xyzList0
        print "The xyzList1 is:", xyzList1



        # If the user push "ESC" the program close down.
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            i.closeDown()
            break
    print "The camera is not open...."

if __name__ == '__main__':
    main()