#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import classes from component files
from Input import Input
from Preprocessing import Preprocessing
from Plot3D import Plot3D

# Import other libraries
import cv2
import numpy as np

def main():

    saveToImagePath = "/home/christian/workspace_python/MasterThesis/SemiSupervisedLearning/writefiles/"
    readFromImagePath = "/home/christian/workspace_python/MasterThesis/SemiSupervisedLearning/readfiles/"

    # Initialize the classes, and load the RGB images of training data class1,2,3 and testing class0.
    i = Input(saveToImagePath, readFromImagePath)
    p = Preprocessing(saveToImagePath, readFromImagePath)
    p3D = Plot3D(saveToImagePath, readFromImagePath)

    # print "The size of the trainingdata class 2 is:", i.trainingData2.shape
    # cv2.imshow("The trainingData2 looks like this", i.trainingData2)
    # cv2.waitKey(0)

    # Now we make a funtion, that runs through the RGB channel together and look for black pixels
    # A black pixels in a RGB image is where the R,B and B is all zero
    # A gray pixel in a RGB image, is where the R,G, and B is all 128
    # A white pixel in a RGB image, is where the R,G, and B is all 255

    listOfBackGroundPixels, listOfSeedPixels, listOfSproutPixels, listOfNolabelledPixels = \
        p.GetBackGroundSeedAndSproutPixels(i.trainingData1, i.trainingData2)

    # Try to take the list and make it to a matrix and show this as an image.
    # First how is the structure of a image?
    backGroundImage = p.getImageFromList(listOfBackGroundPixels)
    seedImage = p.getImageFromList(listOfSeedPixels)
    sproutImage = p.getImageFromList(listOfSproutPixels)
    nolabelledImage = p.getImageFromList(listOfNolabelledPixels)

    # Try to get a 3D plot of all the samples from backGroundImage, seedImage, sproutImage
    # p3D.plot3Dpoints(backGroundImage, seedImage, sproutImage, nolabelledImage)

    cv2.imshow("backGroundImage", backGroundImage)
    cv2.imshow("seedImage", seedImage)
    cv2.imshow("sproutImage", sproutImage)
    cv2.imshow("nolabelledImage", nolabelledImage)
    cv2.imwrite(saveToImagePath + "backGroundImage.png", backGroundImage)
    cv2.imwrite(saveToImagePath + "seedImage.png", seedImage)
    cv2.imwrite(saveToImagePath + "sproutImage.png", sproutImage)
    cv2.imwrite(saveToImagePath + "nolabelledImage.png", nolabelledImage)

    cv2.waitKey(0)

    print "Program ended..."

if __name__ == '__main__':
    main()