#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Preprocessing(object):

    def __init__(self, saveToImagePath, readFromImagePath):
        self.saveToImagePath = saveToImagePath
        self.readFromImagePath = readFromImagePath

    def GetBackGroundSeedAndSproutPixels(self, imgOriginalRGB, imgDrawingRGB):
        listOfBackGroundPixels = []
        listOfSeedPixels = []
        listOfSproutPixels = []
        listOfNolabelledPixels = []

        # Show the image, using PIL
        # imgDrawingRGB.show()

        # Get the cols and rows for the input image
        cols, rows = imgDrawingRGB.size

        pixelimgDrawingRGB = imgDrawingRGB.load()
        pixelimgOriginalRGB = imgOriginalRGB.load()

        # Find all the black pixels in the imgDrawingRGB by drawing them RED f.eks.
        for y in range(rows):
            for x in range(cols):
                if pixelimgDrawingRGB[x, y] == (0, 0, 0):
                    listOfBackGroundPixels.append(pixelimgOriginalRGB[x, y])
                elif pixelimgDrawingRGB[x, y] == (128, 128, 128):
                    listOfSeedPixels.append(pixelimgOriginalRGB[x, y])
                elif pixelimgDrawingRGB[x, y] == (255, 255, 255):
                    listOfSproutPixels.append(pixelimgOriginalRGB[x, y])
                else:
                    listOfNolabelledPixels.append(pixelimgOriginalRGB[x, y])
        return listOfBackGroundPixels, listOfSeedPixels, listOfSproutPixels, listOfNolabelledPixels

    def getImageFromList(self, listOfPixels):
        squareLength = int(np.ceil(np.sqrt(len(listOfPixels))))
        tempImg = np.zeros((squareLength, squareLength, 3), np.uint8)

        # Fill each pixel with the values in listOfPixels
        index = 0
        for x in range(squareLength):
            for y in range(squareLength):
                tempImg[x,y] = listOfPixels[index]

                # If we get to the last element, we break this nested for loop and let the rest of the pixel be black...
                if index == len(listOfPixels)-1:
                    break
                index = index + 1

        # From here we cut of the black pixels, that was not touched, so we dont have an image with black pixels.
        # Note a test of the sprout, there was 30 black pixels left. However the diffecen showed 31 so I subtracted with 1
        numberOfLeftBlackPixels = squareLength*squareLength - len(listOfPixels)-1

        # Cropping the lower line out, if there is any black pixels left.
        numberOfRemovingLowerLine = int(np.ceil(numberOfLeftBlackPixels/float(squareLength)))

        # Making a ROI img[rowStart:rowEnd, colStart:colEnd]
        tempImg = tempImg[0:squareLength-numberOfRemovingLowerLine, 0:squareLength, :]

        # Split the BGR and merge them as RGB. (Kind of slow way to do it..., but i do not know how to swop channels jet)
        b, g, r = cv2.split(tempImg)
        tempImg = cv2.merge((r, g, b))

        return tempImg
