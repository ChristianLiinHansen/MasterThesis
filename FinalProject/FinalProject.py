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
    saveImages = True
    showImages = False   # However the classified featureplot and final classification is still showed...
    saveImagePath = "/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/"

    # Save the input images for documentation
    if saveImages:
        cv2.imwrite(saveImagePath + "trainingData1.png", i.trainingData1)
        cv2.imwrite(saveImagePath + "trainingData2.png", i.trainingData2)
        cv2.imwrite(saveImagePath + "trainingData3.png", i.trainingData3)

    # Test the input images. This was OK at 5/5-2015
    if showImages:
        cv2.imshow("The trainingData1 image", i.trainingData1)
        cv2.imshow("The trainingData2 image", i.trainingData2)
        cv2.imshow("The trainingData3 image", i.trainingData3)

    # Initialize the Preprocessing component with the training data1, 2, 3
    # Doing the 3 classes classificaiotn
    p1 = Preprocessing(i.trainingData1, 1)
    p2 = Preprocessing(i.trainingData2, 2)
    p3 = Preprocessing(i.trainingData3, 3)

    # Save the front ground images for documentation for the training data
    # Only outcomment this, if we change training data.
    if saveImages:
        cv2.imwrite(saveImagePath + "imgFrontGround1.png", p1.imgFrontGround)
        cv2.imwrite(saveImagePath + "imgFrontGround2.png", p2.imgFrontGround)
        cv2.imwrite(saveImagePath + "imgFrontGround3.png", p3.imgFrontGround)
        # Save the sprout images for documentation
        cv2.imwrite(saveImagePath + "imgSprout1.png", p1.imgSprout)
        cv2.imwrite(saveImagePath + "imgSprout2.png", p2.imgSprout)
        cv2.imwrite(saveImagePath + "imgSprout3.png", p3.imgSprout)
        # Save the seed and sprout images for documentation
        cv2.imwrite(saveImagePath + "imgSeedAndSprout1.png", p1.imgSeedAndSprout)
        cv2.imwrite(saveImagePath + "imgSeedAndSprout2.png", p2.imgSeedAndSprout)
        cv2.imwrite(saveImagePath + "imgSeedAndSprout3.png", p3.imgSeedAndSprout)

    # Test the preprocessing images with 3 classes
    if showImages:
        cv2.imshow("trainingData1 sprout image", p1.imgSeedAndSprout)
        cv2.imshow("trainingData2 sprout image", p2.imgSeedAndSprout)
        cv2.imshow("trainingData3 sprout image", p3.imgSeedAndSprout)

    # Initializing the Segmentation component with 3 clases.
    s1 = Segmentation(i.trainingData1, p1.imgFrontGround, p1.imgSeedAndSprout, p1.imgSprout, 1)
    s2 = Segmentation(i.trainingData2, p2.imgFrontGround, p2.imgSeedAndSprout, p2.imgSprout, 2)
    s3 = Segmentation(i.trainingData3, p3.imgFrontGround, p3.imgSeedAndSprout, p3.imgSprout, 3)

    # Only outcomment this, if we change training data.
    # Save the contour images for documentation
    if saveImages:
        cv2.imwrite(saveImagePath + "imgContours1.png", s1.imgContours)
        cv2.imwrite(saveImagePath + "imgContours2.png", s2.imgContours)
        cv2.imwrite(saveImagePath + "imgContours3.png", s3.imgContours)
        # Save the drawing of sprout bounding boxes
        cv2.imwrite(saveImagePath + "imgDraw1.png", s1.imgDraw)
        cv2.imwrite(saveImagePath + "imgDraw2.png", s2.imgDraw)
        cv2.imwrite(saveImagePath + "imgDraw3.png", s3.imgDraw)

    # Test the segmentation with 3 classes
    if showImages:
        cv2.imshow("TrainingData1 contours", s1.imgContours)
        cv2.imshow("TrainingData2 contours", s2.imgContours)
        cv2.imshow("TrainingData3 contours", s3.imgContours)
        cv2.imshow("TrainingData1 imgDraw", s1.imgDraw)
        cv2.imshow("TrainingData2 imgDraw", s2.imgDraw)
        cv2.imshow("TrainingData3 imgDraw", s3.imgDraw)

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
                       showImages)

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

        # Clear the trackbar setting window, since we only want to look at the final classification image
        cv2.destroyWindow(nameOfVideoStreamWindow)

        # Input from webcamera - Testing data
        imgInput = i.getCroppedImg()

        # The input image is processed through each component as followed, with class 0, since it is unknow which class the
        # test image belogns to...
        p = Preprocessing(imgInput, 0)

        # The FrontGround image and SeedAndSprout image is used in the segmentation component
        s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedAndSprout, p.imgSprout, 0)

        if showImages:
            # Plot the featureplot for the testing data, e.i class 0
            featureplotClass0 = PlotFigures(3, "", "")
            featureplotClass0.clearFigure() # In order to have a "live" image we clear all information and plot it again
            featureplotClass0.fig.suptitle("Testing data \n" + str(len(s.contoursFrontGroundFiltered)) + " samples", fontsize=22, fontweight='normal')
            featureplotClass0.plotData(s.featureLengthList, s.featureNumberOfSproutPixelsList, "gs", "class 0")
            featureplotClass0.setXlabel(c.Xlabel)
            featureplotClass0.setYlabel(c.Ylabel)
            featureplotClass0.limit_x(0, c.maxX)
            featureplotClass0.limit_y(0, c.maxY)
            featureplotClass0.addLegend()
            featureplotClass0.updateFigure()
            featureplotClass0.saveFigure("TestingData")

        # Now with the featureplot of class0, we need to draw the featureplot where the class0 is going to get classified.
        # I.e. we want an plot, where the same testing data is seperated into red or blue area, like the training data.
        featureplotClass0Classified = PlotFigures(4, "Feature plot for classified test data", "test")
        featureplotClass0Classified.clearFigure()

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

        # # Here we plot the data that has been classified with 3 classes
        featureplotClass0Classified.fig.suptitle("Feature plot for classified test data \n "+ str(len(s.contoursFrontGroundFiltered)) + " samples", fontsize=22, fontweight='normal')
        featureplotClass0Classified.plotData(featureClass1ListX, featureClass1ListY, "bs", "class 1")
        featureplotClass0Classified.plotData(featureClass2ListX, featureClass2ListY, "rs", "class 2")
        featureplotClass0Classified.plotData(featureClass3ListX, featureClass3ListY, "ys", "class 3")
        featureplotClass0Classified.plotContourf(c.xx, c.yy, c.Z)
        featureplotClass0Classified.limit_x(0, c.maxX)
        featureplotClass0Classified.limit_y(0, c.maxY)
        featureplotClass0Classified.addLegend()
        featureplotClass0Classified.setXlabel(c.Xlabel)
        featureplotClass0Classified.setYlabel(c.Ylabel)
        featureplotClass0Classified.updateFigure()
        featureplotClass0Classified.saveFigure("FeaturePlotForClassifiedTestData")

        # Show the final result...
        cv2.imshow("Show the classified result", c.imgClassified)

        if saveImages:
            # Saving image from input component
            cv2.imwrite(saveImagePath + "imgInput.png", imgInput)
            # Saving image from preprocessing componet
            cv2.imwrite(saveImagePath + "imgFrontGround0.png", p.imgFrontGround)
            cv2.imwrite(saveImagePath + "imgSprout0.png", p.imgSprout)
            cv2.imwrite(saveImagePath + "imgSeedAndSprout0.png", p.imgSeedAndSprout)
            # Saving image from segmentation component
            cv2.imwrite(saveImagePath + "imgDraw0.png", s.imgDraw)
            cv2.imwrite(saveImagePath + "imgContours0.png", s.imgContours)
            # Saving image from classification component
            cv2.imwrite(saveImagePath + "imgClassified.png", c.imgClassified)

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