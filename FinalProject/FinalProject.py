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

    # Test the input images. This was OK at 5/5-2015
    # cv2.imshow("The input RGB image from camera", i.testingData)
    # cv2.imshow("The trainingData1 image", i.trainingData1)
    # cv2.imshow("The trainingDataNeg1 image", i.trainingDataNeg1)
    # cv2.waitKey(0)

    # Initialize the Preprocessing component with the training data1 and -1
    # Input: trainingData1, trainingDataNeg1
    # Output: imgThreshold, imgSeedAndSprout.
    # p1 = Preprocessing(i.trainingData1, 1)
    # pNeg1 = Preprocessing(i.trainingDataNeg1, -1)

    # Doing the 3 classes classificaiotn
    p1 = Preprocessing(i.trainingData1, 1)
    p2 = Preprocessing(i.trainingData2, 2)
    p3 = Preprocessing(i.trainingData3, 3)

    # Test the preprocessing images with 2 classes. This was OK at 5/5-2015
    # cv2.imshow("trainingData1 frontground image", p1.imgFrontGround)
    # cv2.imshow("trainingData1 sprout image", p1.imgSprout)
    # cv2.imshow("trainingData1 seed and sprout image", p1.imgSeedAndSprout)
    # cv2.imshow("trainingDataNeg1 frontground image", pNeg1.imgFrontGround)
    # cv2.imshow("trainingDataNeg1 sprout image", pNeg1.imgSprout)
    # cv2.imshow("trainingDataNeg1 seed and sprout image", pNeg1.imgSeedAndSprout)
    # cv2.waitKey(0)

    # Test the preprocessing images with 3 classes
    # cv2.imshow("trainingData1 sprout image", p1.imgSeedAndSprout)
    # cv2.imshow("trainingData2 sprout image", p2.imgSeedAndSprout)
    # cv2.imshow("trainingData3 sprout image", p3.imgSeedAndSprout)
    # cv2.waitKey(0)

    # Initializing the Segmentation component with 2 clases.
    # s1 = Segmentation(i.trainingData1, p1.imgFrontGround, p1.imgSeedAndSprout, p1.imgSprout, 1)
    # sNeg1 = Segmentation(i.trainingDataNeg1, pNeg1.imgFrontGround, pNeg1.imgSeedAndSprout, pNeg1.imgSprout, -1)

    # Initializing the Segmentation component with 3 clases.
    s1 = Segmentation(i.trainingData1, p1.imgFrontGround, p1.imgSeedAndSprout, p1.imgSprout, 1)
    s2 = Segmentation(i.trainingData2, p2.imgFrontGround, p2.imgSeedAndSprout, p2.imgSprout, 2)
    s3 = Segmentation(i.trainingData3, p3.imgFrontGround, p3.imgSeedAndSprout, p3.imgSprout, 3)

    # Test the segmentation images with 2 classes
    # cv2.imshow("TrainingData1 contours", s1.imgContours)
    # cv2.imshow("TrainingData1 imgDraw", s1.imgDraw)
    # cv2.imshow("TrainingDataNeg1 contours", sNeg1.imgContours)
    # cv2.imshow("TrainingDataNeg1 imgDraw ", sNeg1.imgDraw)
    # cv2.waitKey(0)

    # Test the segmentation with 3 classes
    # cv2.imshow("TrainingData1 contours", s1.imgContours)
    # cv2.imshow("TrainingData2 contours", s2.imgContours)
    # cv2.imshow("TrainingData3 contours", s3.imgContours)
    # cv2.imshow("TrainingData1 imgDraw", s1.imgDraw)
    # cv2.imshow("TrainingData2 imgDraw", s2.imgDraw)
    # cv2.imshow("TrainingData3 imgDraw", s3.imgDraw)
    # cv2.waitKey(0)

    # print "The s1.featureLengthList is:", s1.featureLengthList
    # print "The s1.featureNumberOfSproutPixelsList is:", s1.featureNumberOfSproutPixelsList
    # print "The s1.featureClassStampList is:", s1.featureClassStampList
    # print "The sNeg1.featureLengthList is:", sNeg1.featureLengthList
    # print "The sNeg1.featureNumberOfSproutPixelsList is:", sNeg1.featureNumberOfSproutPixelsList
    # print "The sNeg1.featureClassStampList is:", sNeg1.featureClassStampList

    # # Initialize the Classification component with 2 classes
    # c = Classification(s1.featureLengthList,
    #                    s1.featureNumberOfSproutPixelsList,
    #                    s1.featureClassStampList,
    #                    sNeg1.featureLengthList,
    #                    sNeg1.featureNumberOfSproutPixelsList,
    #                    sNeg1.featureClassStampList,
    #                    False)

    # Initialize the Classification component with 3 classes
    c = Classification(s1.featureLengthList,
                       s1.featureNumberOfSproutPixelsList,
                       s1.featureClassStampList,
                       s2.featureLengthList,
                       s2.featureNumberOfSproutPixelsList,
                       s2.featureClassStampList,
                       s3.featureLengthList,
                       s3.featureNumberOfSproutPixelsList,
                       s3.featureClassStampList,
                       True)



    # Initialize the Output component
    o = Output()

    # At this point, the whole system has been taught with supervised learning.
    # Training data has been loaded, preprocessed, segmented, feature extracted and classified.
    # From here, the testing data is loaded by using the webcam, where each seed will be preprocessed, segmented and classified
    # based on what how the line of seperation lies.

    userCloseDown = False

    # Setting the names of different windows
    nameOfTrackBarWindow = "Trackbar settings"
    nameOfTrackBar1 = "Start system"
    nameOfTrackBar2 = "Absolute exposure"
    nameOfTrackBar3 = "Sharpness"
    nameOfTrackBar4 = "Absolute focus"
    nameOfVideoStreamWindow = "Trackbar settings" # I just set the trackbar and the streaming video in the same window...

    # Add the trackbar in the trackbar window
    i.addTrackbar(nameOfTrackBar1, nameOfTrackBarWindow, i.buttonStartSystem, 1)
    i.addTrackbar(nameOfTrackBar2, nameOfTrackBarWindow, i.absoluteExposure, 2047)
    i.addTrackbar(nameOfTrackBar3, nameOfTrackBarWindow, i.sharpness, 255)
    i.addTrackbar(nameOfTrackBar4, nameOfTrackBarWindow, i.absoluteFocus, 255)

    while i.cameraIsOpen:

        # If the user has not pushed the start button.
        while not i.buttonStartSystem:
            # If the user push "ESC" the program close down.
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                userCloseDown = True
                i.closeDown()
                cv2.destroyWindow(nameOfVideoStreamWindow)
                break

            # Listen for changes for adjusting the settings
            i.startTrackBar(nameOfTrackBar1, nameOfTrackBarWindow)
            i.absoluteExposureTrackBar(nameOfTrackBar2, nameOfTrackBarWindow)
            i.sharpnessTrackBar(nameOfTrackBar3, nameOfTrackBarWindow)
            i.absoluteFocusTrackBar(nameOfTrackBar4, nameOfTrackBarWindow)

            # Then do the adjustments and call the v4l2 settings.
            i.setV4L2(i.absoluteFocus, i.absoluteExposure, i.sharpness)

            # Show the result afterwards.
            cv2.imshow(nameOfVideoStreamWindow, i.getCroppedImg())
            cv2.waitKey(1) # Needs to have this, otherwise I cant see the VideoStreamingWindow with the trackbars

        # if user wants to close down the program, we do it..
        if userCloseDown:
            break

        # Input from webcamera - Testing data
        # imgInput = i.getImg()

        imgInput = i.getCroppedImg()
        cv2.imshow(nameOfVideoStreamWindow, imgInput)
        # Here the directly loaded image needs to be cropped, and adjusted in a way so the test data is the same format

        # # As a beginning, the testing data is for now, just a still image, with a mix of diffrent seeds
        # # Later the imgInput should come from the camera as written above.
        # imgInput = i.testingData

        # The input image is processed through each component as followed, with class 0, since it is unknow which class the
        # test image belogns to...
        # print "Preprocessing image..."
        p = Preprocessing(imgInput, 0)
        # print "Done preprocessing image..."

        # Show the output of the testData
        processingPath = "/home/christian/Dropbox/E14/Master-thesis-doc/images/Section6/TestingInRoboLab/4_5_2015/"
        # cv2.imshow("The RGB input image", imgInput)
        # cv2.imshow("The front ground image", p.imgFrontGround)
        # cv2.imshow("The imgSprout image", p.imgSprout)
        # cv2.imshow("The imgSeedAndSprout image", p.imgSeedAndSprout)
        # cv2.imwrite(processingPath + "imgInput.png", imgInput)
        # cv2.imwrite(processingPath + "imgFrontGround.png", p.imgFrontGround)
        # cv2.imwrite(processingPath + "imgSprout.png", p.imgSprout)
        # cv2.imwrite(processingPath + "imgSeedAndSprout.png", p.imgSeedAndSprout)

        # The FrontGround image and SeedAndSprout image is used in the segmentation component
        # print "Segmentate image..."
        s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedAndSprout, p.imgSprout, 0)
        cv2.imshow("Show the RGB image with contours of sprouts", s.imgDraw)
        # print "Done segmentate image..."

        if False:
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

        # print "The x feature list is:", s.featureLengthList
        # print "The y feature list is:", s.featureNumberOfSproutPixelsList
        # We combine the x,y feature list into a single list with (x,y) points.

        # Getting the featureplot with 2 classes
        # featureClass1ListX, \
        # featureClass1ListY, \
        # featureClassNeg1ListX, \
        # featureClassNeg1ListY, \
        # centerClass1List, \
        # centerClassNeg1List \
        #     = c.getClassifiedLists(s.featureLengthList, s.featureNumberOfSproutPixelsList, s.featureCenterOfMassList, s.imgRGB)

        # Getting the featureplot with 3 classes
        featureClass1ListX, \
        featureClass1ListY, \
        centerClass1List, \
        featureClass2ListX, \
        featureClass2ListY, \
        centerClass2List, \
        featureClass3ListX, \
        featureClass3ListY, \
        centerClass3List = c.getClassifiedLists3classes(s.featureLengthList, s.featureNumberOfSproutPixelsList, s.featureCenterOfMassList, s.imgRGB)

        # # Here we plot the data that has been classified with 2 classes
        # featureplotClass0Classified.plotData(featureClass1ListX, featureClass1ListY, "rs", "class 1")
        # featureplotClass0Classified.plotData(featureClassNeg1ListX, featureClassNeg1ListY, "bs", "class -1")
        # featureplotClass0Classified.plotContourf(c.xx, c.yy, c.Z)
        # featureplotClass0Classified.limit_x(0, c.maxX)
        # featureplotClass0Classified.limit_y(0, c.maxY)
        # featureplotClass0Classified.setTitle("Feature plot for classified test data")
        # featureplotClass0Classified.addLegend()
        # featureplotClass0Classified.setXlabel(c.Xlabel)
        # featureplotClass0Classified.setYlabel(c.Ylabel)
        # featureplotClass0Classified.updateFigure()

        # # Here we plot the data that has been classified with 3 classes
        featureplotClass0Classified.plotData(featureClass1ListX, featureClass1ListY, "rs", "class 1")
        featureplotClass0Classified.plotData(featureClass2ListX, featureClass2ListY, "bs", "class 2")
        featureplotClass0Classified.plotData(featureClass3ListX, featureClass3ListY, "gs", "class 3")
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
        # xyzList0, xyzList1 = o.convertUV2XYZ(centerClass1List, centerClassNeg1List, imgInput.shape)
        # print "The xyzList0 is:", xyzList0
        # print "The xyzList1 is:", xyzList1

        # The xyzList0 contains the list with blue coordinates
        # The xyzList1 contains the list with red coordinates

        # If the user push "ESC" the program close down.
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            userCloseDown = True
            i.closeDown()
            break
    if userCloseDown:
        print "User closed the program..."
    else:
        print "The camera is not open.... "

if __name__ == '__main__':
    main()