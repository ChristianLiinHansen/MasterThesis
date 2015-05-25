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
    showAndSaveImagesFlag = True  # However the classified featureplot and final classification is still showed...
    normalization = True # Showing normalization data
    saveImagePath = "/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/"

    # Initialize the Preprocessing component with the training data1, 2, 3
    p1 = Preprocessing(i.trainingData1, 1, saveImagePath)
    p2 = Preprocessing(i.trainingData2, 2, saveImagePath)
    p3 = Preprocessing(i.trainingData3, 3, saveImagePath)

    # Here the trainingdata for class1, class2, and class3 has been trimmed.
    # i.e. the data that should be loaded into the segmentation component is the following images
    # imgSproutClass1 = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/imgSproutClass1WithMorph.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # imgSproutClass2 = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/imgSproutClass2WithMorph.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # imgSproutClass3 = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/imgSproutClass3WithMorph.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # imgSeedAndSproutClass1 = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/imgSeedAndSproutClass1WithMorph.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # imgSeedAndSproutClass2 = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/imgSeedAndSproutClass2WithMorph.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # imgSeedAndSproutClass3 = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/imgSeedAndSproutClass3WithMorph.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # Initializing the Segmentation component with 3 clases.
    # Using global HSV setting
    s1 = Segmentation(i.trainingData1, p1.imgFrontGround, p1.imgSeedAndSprout, p1.imgSprout, 1, saveImagePath)
    s2 = Segmentation(i.trainingData2, p2.imgFrontGround, p2.imgSeedandSproutRepaired, p2.imgSproutRepaired, 2, saveImagePath)
    s3 = Segmentation(i.trainingData3, p3.imgFrontGround, p3.imgSeedandSproutRepaired, p3.imgSproutRepaired, 3, saveImagePath)

    # Using local HSV setting, i.e. loading imgSeedAndSprout and imgSprout which has individuelly been trimmed to have better sprouts
    # s1 = Segmentation(i.trainingData1, p1.imgFrontGround, imgSeedAndSproutClass1, imgSproutClass1, 1)
    # s2 = Segmentation(i.trainingData2, p2.imgFrontGround, imgSeedAndSproutClass2, imgSproutClass2, 2)
    # s3 = Segmentation(i.trainingData3, p3.imgFrontGround, imgSeedAndSproutClass3, imgSproutClass3, 3)

    # Choise which feature to use:
    # featureCenterOfMassList,                   # feature 0
    # featureLengthList,                         # feature 1
    # featureWidthList,                          # feature 2
    # featureRatioList,                          # feature 3
    # featureNumberOfSproutPixelsList,           # feature 4
    # featureHueMeanList,                        # feature 5
    # featureHueStdList,                         # feature 6
    # featureClassStampList                      # feature 7
    featureIndexX = 1
    featureIndexY = 4

    # Initialize the clasification component for the training data
    c = Classification(s1.listOfFeatures, s2.listOfFeatures, s3.listOfFeatures, featureIndexX, featureIndexY, showAndSaveImagesFlag, saveImagePath, normalization)

    # Initialize the Output component
    o = Output()

    # At this point, the whole system has been taught with supervised learning.
    # Training data has been loaded, preprocessed, segmented, feature extracted and classified.
    # From here, the testing data is loaded by using the webcam, where each seed will be preprocessed, segmented and classified
    # based on what how the line of seperation lies.

    userCloseDown = False

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

    # Save the input images for documentation
    if showAndSaveImagesFlag:
        print "The showAndSaveImagesFlag is true"
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

    while i.cameraIsOpen:
        print "Camera is open..."
        # If the user has not pushed the start button.
        while not i.buttonStartSystem:
            # If the user push "ESC" the program close down.
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                userCloseDown = True
                i.closeDown()
                cv2.destroyWindow(nameOfTrackBarWindowRGB)
                cv2.destroyWindow(nameOfVideoStreamWindowHSVclass1)
                break

            # Listen for changes for adjusting the settings in the trackbar setting RGB window
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

            ############################################################
            # Trying to trim the training data as much as possible
            ############################################################
            # Using the trackbars from the trackbar setting HSV window to adjust the HSV segmented image, i.e. sprout image
            # imgSproutClassX = i.getSprout(i.hueMin, i.hueMax, i.saturationMin, i.saturationMax, i.valueMin, i.valueMax)

            # Do some morph about it... Der sker et offset i billedet,
            # hvis jeg kalder getClosing fra Preprocessing classen,
            # da jeg i forvejen laver dette offset fra klassen
            # kernel = np.matrix(([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), np.uint8)
            # imgSproutClassX = cv2.dilate(imgSproutClassX, kernel, iterations=3)
            # imgSproutClassX = cv2.erode(imgSproutClassX, kernel, iterations=3)
            # imgSproutClassX = cv2.erode(imgSproutClassX, kernel, iterations=1)
            # imgSproutClassX = cv2.dilate(imgSproutClassX, kernel, iterations=1)

            # Adding the new sprout image to the old front ground image, since the front ground image is OK
            # imgSeedAndSproutClassX = cv2.add(imgSproutClassX, p3.imgFrontGround)

            # Get the contours
            # contoursFrontGroundClassX = s1.getContours(p1.imgFrontGround)
            # contoursFrontGroundFilteredClassX,a,b,c = s1.getContoursFilter(contoursFrontGroundClassX, 200, 4000)

            # Use the segmentation part to get the contours, to get the boundingbox
            # of the new imgSeedAndSprout image for class 2
            # s3.getFeaturesFromEachROI(contoursFrontGroundFilteredClassX, imgSeedAndSproutClassX, imgSproutClassX, i.trainingData3, 3)
            # cv2.imshow("BoundingBox image of classX", s3.imgDraw)

            # Take the black/white image and convert it to color, so we can draw some green seed bounderies on it
            # imgSproutWithBounderyClassX = cv2.cvtColor(imgSproutClassX, cv2.COLOR_GRAY2BGR)
            # Add the drawn contour image together with the black/white image, which now can be have color on it
            # imgSproutWithBounderyClassX = cv2.add(imgSproutWithBounderyClassX, s3.imgContours)

            # Then do the adjustments and call the v4l2 settings.
            i.setV4L2(i.absoluteFocus, i.absoluteExposure, i.sharpness)

            # Show the result afterwards.
            cv2.imshow(nameOfTrackBarWindowRGB, i.getCroppedImg())
            # cv2.imwrite(saveImagePath + "imgRGBwith4000K.png", i.getCroppedImg())

            # NOTE!!!: This window is the class1 sprout image. I.e. it is only the processed image of the training data, not the image from webcamera
            # cv2.imshow(nameOfVideoStreamWindowHSVclass1, imgSproutWithBounderyClassX)

            cv2.waitKey(1) # Needs to have this, otherwise I cant see the VideoStreamingWindow with the trackbars

        # if user wants to close down the program, we do it..
        if userCloseDown:
            break

        # Clear the trackbar setting window, since we only want to look at the final classification image
        cv2.destroyWindow(nameOfTrackBarWindowRGB)
        cv2.destroyWindow(nameOfVideoStreamWindowHSVclass1)

        # Input from webcamera - Testing data
        # imgInput = i.getCroppedImg()
        imgInput = i.testingData # Using a still test image, when the real USB camera is not available

        # The input image is processed through each component as followed, with class 0, since it is unknow which class the
        # test image belogns to...
        p = Preprocessing(imgInput, 0, saveImagePath)

        # The FrontGround image and SeedAndSprout image is used in the segmentation component
        # s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedAndSprout, p.imgSprout, 0)
        s = Segmentation(imgInput, p.imgFrontGround, p.imgSeedandSproutRepaired, p.imgSproutRepaired, 0, saveImagePath)

        if False:
            # Plot the featureplot for the testing data, e.i class 0
            featureplotClass0 = PlotFigures(3, "Feature plot for testing data class 0 with" + str(len(s.listOfFeatures[featureIndexX])) + " samples", "", saveImagePath)
            featureplotClass0.clearFigure() # In order to have a "live" image we clear all information and plot it again
            featureplotClass0.fig.suptitle("Feature plot for testing data with \n" + str(len(s.listOfFeatures[featureIndexX])) + " samples", fontsize=22, fontweight='normal')
            featureplotClass0.plotData(c.NormalizeData(s.listOfFeatures[featureIndexX]), c.NormalizeData(s.listOfFeatures[featureIndexY]), "gs", "class 0")
            featureplotClass0.setXlabel(c.Xlabel)
            featureplotClass0.setYlabel(c.Ylabel)
            featureplotClass0.limit_x(0, c.maxX)
            featureplotClass0.limit_y(0, c.maxY)
            featureplotClass0.addLegend()
            featureplotClass0.updateFigure()
            featureplotClass0.saveFigure("TestingData")

        # Now with the featureplot of class0, we need to draw the featureplot where the class0 is going to get classified.
        # I.e. we want an plot, where the same testing data is seperated into red or blue area, like the training data.
        featureplotClass0Classified = PlotFigures(4, "Feature plot for classified test data", "test", saveImagePath)
        featureplotClass0Classified.clearFigure()
        # x = 0.4
        # y = 0.2
        # featureplotClass0.plotMean(x, y, "cs")
        # x = int(x * 1/c.h)
        # y = int(y * 1/c.h)
        # print "So at x: (", x, ") and y: (", y, ") the Z-value is:", c.Z[y-1, x-1], "\n"
        # plt.show(block=False)   # It is very big with 300 dpi
        # plt.draw()

        # Is th training data ok?
        # Well the training data have some false positives, but the SVM is not overfitting
        print "Now we are at the end of the program..."
        # cv2.imshow("The class 1 data", s1.imgDraw)
        # cv2.imshow("The class 2 data", s2.imgDraw)
        # cv2.imshow("The class 3 data", s3.imgDraw)

        # However the testingdata (which is right now also a still image)
        # might needs to be look at.

        # cv2.imshow("Testing data input", i.testingData)
        # cv2.imshow("Preprocessing stuff1", p.imgSeedAndSprout)
        # cv2.imshow("Preprocessing stuff2", p.imgSeedandSproutRepaired)
        # cv2.imshow("Segmentation stuff1", s.imgContours)
        # cv2.imshow("Segmentation stuff2", s.imgDraw)

        # Save the images
        # cv2.imwrite(saveImagePath + "imgDrawClass1.png", s1.imgDraw)
        # cv2.imwrite(saveImagePath + "imgDrawClass2.png", s2.imgDraw)
        # cv2.imwrite(saveImagePath + "imgDrawClass3.png", s3.imgDraw)
        #
        # cv2.imwrite(saveImagePath + "imgSeedAndSproutClass0.png", p.imgSeedAndSprout)
        # cv2.imwrite(saveImagePath + "imgSeedandSproutRepairedClass0.png", p.imgSeedandSproutRepaired)
        # cv2.imwrite(saveImagePath + "imgContoursClass0.png", s.imgContours)
        # cv2.imwrite(saveImagePath + "imgDrawClass0.png", s.imgDraw)

        # cv2.imshow("The imgSeedandSproutRepaired  image", p.imgSeedandSproutRepaired)
        # cv2.imshow("The imgSeedAndSprout  image", p.imgSeedAndSprout)
        # cv2.imshow("The segmentation image", s.imgDraw)

        # imgPause = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/3Classes/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        # cv2.imshow("Pause program", imgPause)
        # cv2.waitKey(0)
        # return 0

        featureClass1ListX, \
        featureClass1ListY, \
        centerClass1List, \
        featureClass2ListX, \
        featureClass2ListY, \
        centerClass2List, \
        featureClass3ListX, \
        featureClass3ListY, \
        centerClass3List = c.getClassifiedLists3classes(s.listOfFeatures[featureIndexX], s.listOfFeatures[featureIndexY], s.listOfFeatures[0], imgInput)

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

        # Show the segmentation part
        cv2.imshow("Segmentation of testing data", s.imgDraw)

        # Show the final result...
        cv2.imshow("Show the classified result", c.imgClassified)
        cv2.imwrite(saveImagePath + "imgClassified.png", c.imgClassified)

        # imgPause = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/3Classes/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        # cv2.imshow("Pause program", imgPause)
        # cv2.waitKey(0)
        # print "Ending program here -DEBUG"
        # return 0
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