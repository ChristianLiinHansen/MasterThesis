#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import classes from component files
import cv2

from Input import Input
from Preprocessing import Preprocessing
from Segmentation import Segmentation
from Classification import Classification
from Output import Output

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
    # based on what hwo the line of seperation lies.

    while i.cameraIsOpen:

        # Input from webcamera - Testing data
        # imgInput = i.getImg()
        # cv2.imshow("Streaming from camera", imgInput)

        # Showing the training data in order to exit the program...
        # cv2.imshow("TrainingData1", i.trainingData1)
        cv2.imshow("trainingDataNeg1", i.trainingDataNeg1)

        # If the user push "ESC" the program close down.
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            i.closeDown()
            break

if __name__ == '__main__':
    main()