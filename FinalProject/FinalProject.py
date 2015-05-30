#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import classes from component files
import cv2
from Input import Input
from Preprocessing import Preprocessing
from Segmentation import Segmentation
from Classification import Classification
from Output import Output
from PlotFigures import PlotFigures

from Normalize import NormalizeFeature
from pylab import *

def ShowAndSaveTrainingFigures(i, p1,p2,p3,s1,s2,s3, saveImagePath):
    # Show the images in the input component
    cv2.imshow("The trainingData1 RGB image", i.trainingData1)
    cv2.imshow("The trainingData2 RGB image", i.trainingData2)
    cv2.imshow("The trainingData3 RGB image", i.trainingData3)

    # Show the images in the preprocessing component
    cv2.imshow("trainingData1 front ground image", p1.imgFrontGround)
    cv2.imshow("trainingData2 front ground image", p2.imgFrontGround)
    cv2.imshow("trainingData3 front ground image", p3.imgFrontGround)
    cv2.imshow("trainingData1 sprout image", p1.imgSprout)
    cv2.imshow("trainingData2 sprout image", p2.imgSprout)
    cv2.imshow("trainingData3 sprout image", p3.imgSprout)
    cv2.imshow("trainingData1 seed and sprout image", p1.imgSeedAndSprout)
    cv2.imshow("trainingData2 seed and sprout image", p2.imgSeedAndSprout)
    cv2.imshow("trainingData3 seed and sprout image", p3.imgSeedAndSprout)

    # Show the images in the segmentation component
    cv2.imshow("TrainingData1 contours", s1.imgContours)
    cv2.imshow("TrainingData2 contours", s2.imgContours)
    cv2.imshow("TrainingData3 contours", s3.imgContours)
    cv2.imshow("TrainingData1 imgDraw", s1.imgDraw)
    cv2.imshow("TrainingData2 imgDraw", s2.imgDraw)
    cv2.imshow("TrainingData3 imgDraw", s3.imgDraw)

    # Write the images in the input component
    cv2.imwrite(saveImagePath + "trainingData1.png", i.trainingData1)
    cv2.imwrite(saveImagePath + "trainingData2.png", i.trainingData2)
    cv2.imwrite(saveImagePath + "trainingData3.png", i.trainingData3)

    # Write the images in the preprocessing component
    cv2.imwrite(saveImagePath + "imgFrontGround1.png", p1.imgFrontGround)
    cv2.imwrite(saveImagePath + "imgFrontGround2.png", p2.imgFrontGround)
    cv2.imwrite(saveImagePath + "imgFrontGround3.png", p3.imgFrontGround)
    cv2.imwrite(saveImagePath + "imgSprout1.png", p1.imgSprout)
    cv2.imwrite(saveImagePath + "imgSprout2.png", p2.imgSprout)
    cv2.imwrite(saveImagePath + "imgSprout3.png", p3.imgSprout)
    cv2.imwrite(saveImagePath + "imgSeedAndSprout1.png", p1.imgSeedAndSprout)
    cv2.imwrite(saveImagePath + "imgSeedAndSprout2.png", p2.imgSeedAndSprout)
    cv2.imwrite(saveImagePath + "imgSeedAndSprout3.png", p3.imgSeedAndSprout)

    # Write the images in the segmentation components
    cv2.imwrite(saveImagePath + "imgContours1.png", s1.imgContours)
    cv2.imwrite(saveImagePath + "imgContours2.png", s2.imgContours)
    cv2.imwrite(saveImagePath + "imgContours3.png", s3.imgContours)
    cv2.imwrite(saveImagePath + "imgDraw1.png", s1.imgDraw)
    cv2.imwrite(saveImagePath + "imgDraw2.png", s2.imgDraw)
    cv2.imwrite(saveImagePath + "imgDraw3.png", s3.imgDraw)

def TrackBarInit(i):

    # Setting the names of different windows
    nameOfTrackBarWindowRGB = "Trackbar settings RGB"
    nameOfVideoStreamWindowHSVclass1 = "Trackbar settings HSV for training data class1"

    # Setting the names of different parameters in different windows
    nameOfTrackBar1 = "Start system"
    nameOfTrackBar2 = "Absolute exposure"
    nameOfTrackBar3 = "Sharpness"
    nameOfTrackBar4 = "Absolute focus"
    nameOfTrackBar5 = "Hue min"
    nameOfTrackBar6 = "Hue max"
    nameOfTrackBar7 = "Saturation min"
    nameOfTrackBar8 = "Saturation max"
    nameOfTrackBar9 = "Value min"
    nameOfTrackBar10 = "Value max"

    # Add the trackbar in the trackbar window RGB
    i.addTrackbar(nameOfTrackBar1, nameOfTrackBarWindowRGB, i.buttonStartSystem, 1)
    i.addTrackbar(nameOfTrackBar2, nameOfTrackBarWindowRGB, i.absoluteExposure, 2047)
    i.addTrackbar(nameOfTrackBar3, nameOfTrackBarWindowRGB, i.sharpness, 255)
    i.addTrackbar(nameOfTrackBar4, nameOfTrackBarWindowRGB, i.absoluteFocus, 255)

    # Add the trackbar in the trackbar window HSV
    # i.addTrackbar(nameOfTrackBar5, nameOfVideoStreamWindowHSVclass1, i.hueMin, 180)
    # i.addTrackbar(nameOfTrackBar6, nameOfVideoStreamWindowHSVclass1, i.hueMax, 180)
    # i.addTrackbar(nameOfTrackBar7, nameOfVideoStreamWindowHSVclass1, i.saturationMin, 255)
    # i.addTrackbar(nameOfTrackBar8, nameOfVideoStreamWindowHSVclass1, i.saturationMax, 255)
    # i.addTrackbar(nameOfTrackBar9, nameOfVideoStreamWindowHSVclass1, i.valueMin, 255)
    # i.addTrackbar(nameOfTrackBar10, nameOfVideoStreamWindowHSVclass1, i.valueMax, 255)

def TrackBarRun(i):
        # Setting the names of different windows
        nameOfTrackBarWindowRGB = "Trackbar settings RGB"
        nameOfVideoStreamWindowHSVclass1 = "Trackbar settings HSV for training data class1"

        # Setting the names of different parameters in different windows
        nameOfTrackBar1 = "Start system"
        nameOfTrackBar2 = "Absolute exposure"
        nameOfTrackBar3 = "Sharpness"
        nameOfTrackBar4 = "Absolute focus"
        nameOfTrackBar5 = "Hue min"
        nameOfTrackBar6 = "Hue max"
        nameOfTrackBar7 = "Saturation min"
        nameOfTrackBar8 = "Saturation max"
        nameOfTrackBar9 = "Value min"
        nameOfTrackBar10 = "Value max"

        i.startTrackBar(nameOfTrackBar1, nameOfTrackBarWindowRGB)
        i.absoluteExposureTrackBar(nameOfTrackBar2, nameOfTrackBarWindowRGB)
        i.sharpnessTrackBar(nameOfTrackBar3, nameOfTrackBarWindowRGB)
        i.absoluteFocusTrackBar(nameOfTrackBar4, nameOfTrackBarWindowRGB)

        # Listen for changes for adjusting the settings in the trackbar setting HSV window
        i.hueMinTrackBar(nameOfTrackBar5, nameOfVideoStreamWindowHSVclass1)
        i.hueMaxTrackBar(nameOfTrackBar6, nameOfVideoStreamWindowHSVclass1)
        i.saturationMinTrackBar(nameOfTrackBar7, nameOfVideoStreamWindowHSVclass1)
        i.saturationMaxTrackBar(nameOfTrackBar8, nameOfVideoStreamWindowHSVclass1)
        i.valueMinTrackBar(nameOfTrackBar9, nameOfVideoStreamWindowHSVclass1)
        i.valueMaxTrackBar(nameOfTrackBar10, nameOfVideoStreamWindowHSVclass1)

def DestroyWindows():
    nameOfTrackBarWindowRGB = "Trackbar settings RGB"
    nameOfVideoStreamWindowHSVclass1 = "Trackbar settings HSV for training data class1"
    cv2.destroyWindow(nameOfTrackBarWindowRGB)
    cv2.destroyWindow(nameOfVideoStreamWindowHSVclass1)

def ShowFeaturePlotClass0(featureIndexX, featureIndexY, s, c, saveImagePath, normalization):
        # Plot the featureplot for the testing data, e.i class 0
        featureplotClass0 = PlotFigures(3, "Feature plot for testing data class 0 with" + str(len(s.listOfFeatures[featureIndexX])) + " samples", "", saveImagePath)
        featureplotClass0.clearFigure() # In order to have a "live" image we clear all information and plot it again
        featureplotClass0.fig.suptitle("Feature plot for testing data with \n" + str(len(s.listOfFeatures[featureIndexX])) + " samples", fontsize=22, fontweight='normal')

        if normalization:
            featureplotClass0.plotData(c.NormalizeData(s.listOfFeatures[featureIndexX]), c.NormalizeData(s.listOfFeatures[featureIndexY]), "gs", "class 0")
            featureplotClass0.setXlabel(c.Xlabel)
            featureplotClass0.setYlabel(c.Ylabel)
            featureplotClass0.limit_x(0, c.maxX)
            featureplotClass0.limit_y(0, c.maxY)
        else:
            featureplotClass0.plotData(s.listOfFeatures[featureIndexX], s.listOfFeatures[featureIndexY], "gs", "class 0")

        featureplotClass0.setXlabel(c.Xlabel)
        featureplotClass0.setYlabel(c.Ylabel)
        featureplotClass0.addLegend()
        featureplotClass0.updateFigure()
        featureplotClass0.saveFigure("TestingData")

def ShowFeaturePlotClass0Classified(featureClass1ListX, featureClass2ListX, featureClass3ListX, featureClass1ListY, featureClass2ListY, featureClass3ListY,c, saveImagePath):
    # Now with the featureplot of class0, we need to draw the featureplot where the class0 is going to get classified.
            # I.e. we want an plot, where the same testing data is seperated into red or blue area, like the training data.
            featureplotClass0Classified = PlotFigures(4, "Feature plot for classified test data", "test", saveImagePath)
            featureplotClass0Classified.clearFigure()

            # # Here we plot the data that has been classified with 3 classes
            featureplotClass0Classified = PlotFigures(4, "Feature plot for testing data class 1,2,3 \n",
                          "with respectively number of samples: " +
                          str(len(featureClass1ListX)) + "," +
                          str(len(featureClass2ListX)) + "," +
                          str(len(featureClass3ListX)), saveImagePath)
            # featureplotClass0Classified.plotData(c.NormalizeData(s.listOfFeatures[featureIndexX]), c.NormalizeData(s.listOfFeatures[featureIndexY]), "gs", "class 999")
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

def TrackBarStart(i):

    while not i.buttonStartSystem:
        # If the user push "ESC" the program close down.
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            userCloseDown = True
            i.closeDown()
            DestroyWindows()
            break
        # Listen for the trackbars.
        # Listen for changes for adjusting the settings in the trackbar setting RGB window
        TrackBarRun(i)

        # Then do the adjustments and call the v4l2 settings.
        i.setV4L2(i.absoluteFocus, i.absoluteExposure, i.sharpness)

        # Show the result afterwards.
        cv2.imshow("Trackbar settings RGB", i.getCroppedImg())
        # cv2.imwrite(saveImagePath + "imgRGBwith4000K.png", i.getCroppedImg())

        # NOTE!!!: This window is the class1 sprout image. I.e. it is only the processed image of the training data, not the image from webcamera
        # cv2.imshow(nameOfVideoStreamWindowHSVclass1, imgSproutWithBounderyClassX)

        cv2.waitKey(1) # Needs to have this, otherwise I cant see the VideoStreamingWindow with the trackbars

def main():

    # Global control parameters, used for debugging, documentation etc...
    showAndSaveImagesFlag = False  # However the classified featureplot and final classification is still showed...
    normalization = True # Showing normalization data
    vizualize = True
    saveImagePath = "/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/"

    # Initialize the Input component with cameraIndex = 0 (webcamera inbuilt in PC)
    # Input: Plug and play webcamera
    # Output: RGB image, training data and testing data
    i = Input(0)

    # Initialize the Preprocessing component with the training data1, 2, 3
    p1 = Preprocessing(i.trainingData1, 1, saveImagePath)
    p2 = Preprocessing(i.trainingData2, 2, saveImagePath)
    p3 = Preprocessing(i.trainingData3, 3, saveImagePath)

    # Initializing the Segmentation component with 3 clases.
    # Using global HSV setting
    s1 = Segmentation(i.trainingData1, p1.imgFrontGround, p1.imgSeedandSproutRepaired, p1.imgSproutRepaired, 1, saveImagePath)
    s2 = Segmentation(i.trainingData2, p2.imgFrontGround, p2.imgSeedandSproutRepaired, p2.imgSproutRepaired, 2, saveImagePath)
    s3 = Segmentation(i.trainingData3, p3.imgFrontGround, p3.imgSeedandSproutRepaired, p3.imgSproutRepaired, 3, saveImagePath)

    # Choise which feature to use:
    # featureCenterOfMassList,                   # feature 0
    # featureLengthList,                         # feature 1
    # featureWidthList,                          # feature 2
    # featureRatioList,                          # feature 3
    # featureNumberOfSproutPixelsList,           # feature 4
    # featureHueMeanList,                        # feature 5
    # featureHueStdList,                         # feature 6
    # featureClassStampList                      # feature 7
    featureIndexX = 3
    featureIndexY = 4

    # Initialize the clasification component for the training data
    c = Classification(s1.listOfFeatures, s2.listOfFeatures, s3.listOfFeatures, featureIndexX, featureIndexY, vizualize, saveImagePath, normalization)

    # Initialize the Output component
    o = Output()

    # At this point, the whole system has been taught with supervised learning.
    # Training data has been loaded, preprocessed, segmented, feature extracted and classified.
    # From here, the testing data is loaded by using the webcam, where each seed will be preprocessed, segmented and classified
    # based on what how the line of seperation lies.

    userCloseDown = False
    TrackBarInit(i)

    if False:
        ShowAndSaveTrainingFigures(i, p1, p2, p3, s1, s2, s3, saveImagePath)

    # while i.cameraIsOpen: # To avoid beiing depended on the camera or not, we just say the camera is always open.
    # We use still images anayway at the moment...
    while True:
        # print "Camera is open..."
        # If the user has not pushed the start button.
        TrackBarStart(i)

        # if user wants to close down the program, we do it..
        if userCloseDown:
            break

        #############################################################
        # After the training we run in this while loop...
        #############################################################

        # Clear the trackbar setting window, since we only want to look at the final classification image
        DestroyWindows()

        # Input from webcamera - Testing data
        # imgInput = i.getCroppedImg()
        imgInput = i.testingData # Using a still test image, when the real USB camera is not available

        # The input image is processed through each component as followed, with class 0, since it is unknow which class the
        # test image belogns to...
        p = Preprocessing(imgInput, 0, saveImagePath)

        # The FrontGround image and SeedAndSprout image is used in the segmentation component
        # s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedAndSprout, p.imgSprout, 0)
        s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedandSproutRepaired, p.imgSproutRepaired, 0, saveImagePath)

        if vizualize:
            ShowFeaturePlotClass0(featureIndexX, featureIndexY, s, c, saveImagePath, normalization)

        featureClass1ListX, \
        featureClass1ListY, \
        centerClass1List, \
        featureClass2ListX, \
        featureClass2ListY, \
        centerClass2List, \
        featureClass3ListX, \
        featureClass3ListY, \
        centerClass3List = c.getClassifiedLists3classes(s.listOfFeatures[featureIndexX], s.listOfFeatures[featureIndexY], s.listOfFeatures[0], imgInput)

        if vizualize:
            ShowFeaturePlotClass0Classified(featureClass1ListX, featureClass2ListX, featureClass3ListX, featureClass1ListY, featureClass2ListY, featureClass3ListY,c, saveImagePath)

        #############################################
        # Finally we show the result
        ############################################

        cv2.imshow("The final classification", c.imgClassified)
        cv2.imwrite(saveImagePath + "imgClassified.png", c.imgClassified)

        # If the user push "ESC" the program close down.
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            userCloseDown = True
            # i.closeDown()
            break
    if userCloseDown:
        print "User closed the program..."
    else:
        print "The camera is not open.... "

if __name__ == '__main__':
    main()