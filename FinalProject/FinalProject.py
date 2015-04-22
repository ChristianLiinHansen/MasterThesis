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
    i = Input(1)

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

    print "Testing..."
    ion()
    fig1 = figure()
    ax1 = fig1.add_subplot(111)
    x = arange(0,2*pi,0.01)
    y = sin(x)
    line1, = ax1.plot(x, y, 'gs')
    plt.ioff()
    iteration = 0

    # for iteration in arange(1, 200):
    #     line1.set_ydata(sin(x+iteration/10.0))  # update the data
    #     draw()

    featureplot = PlotFigures("Feature space for testing data class 0", "FeatureSpaceClass0")

    while i.cameraIsOpen:

        iteration = iteration + 1
        line1.set_ydata(sin(x+iteration/10.0))  # update the data
        draw()

        # Input from webcamera - Testing data
        # imgInput = i.getImg()
        # cv2.imshow("Streaming from camera", imgInput)

        # # As a beginning, the testing data is for now, just a still image, with a mix of diffrent seeds
        # # Later the imgInput should come from the camera as written above.
        imgInput = i.testingData
        cv2.imshow("Testing data", imgInput)

        # The input image is processed through each component as followed, with class 0, since it is unknow which class the
        # test image belogns to...
        p = Preprocessing(imgInput, 0)

        # The output of the preproceesing step for the test image is as followed:
        # cv2.imshow("Test image imgSeedAndSprout though preprocessing step", p.imgSeedAndSprout)
        # cv2.imshow("Test image imgFrontGround though preprocessing step", p.imgFrontGround)
        # cv2.imshow("Test image imgSprout though preprocessing step", p.imgSprout)

        # The FrontGround image and SeedAndSprout image is used in the segmentation component
        s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedAndSprout, p.imgSprout, 0)
        cv2.imshow("Show the RGB image with contours of sprouts", s.imgDraw)

        # Plot the featureplot for the testing data, e.i class 0
        featureplot.clearFigure()
        featureplot.plotData(s.featureLengthList, s.featureNumberOfSproutPixelsList, "gs", "class 0")
        featureplot.limit_x(0, c.maxX)
        featureplot.limit_y(0, c.maxY)
        featureplot.setTitle("Featureplot for class 0")
        featureplot.addLegend()
        featureplot.setXlabel(c.Xlabel)
        featureplot.setYlabel(c.Ylabel)
        featureplot.updateFigure()

        # featureplot.plotData(s.featureLengthList, s.featureNumberOwfSproutPixelsList, "gs", "class 0")
        # iteration = iteration + 1
        # line1.set_ydata(sin(x+iteration/10.0))  # update the data
        # draw()

        # Showing the training data in order to exit the program...
        # cv2.imshow("TrainingData1", i.trainingData1)
        # cv2.imshow("trainingDataNeg1", i.trainingDataNeg1)


        # If the user push "ESC" the program close down.
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            i.closeDown()
            break

if __name__ == '__main__':
    main()