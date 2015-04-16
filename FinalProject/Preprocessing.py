#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2

class Preprocessing(object):

    def __init__(self, imgInput, classStamp):

        self.classStamp = classStamp
        # Convert the input image to binary image and use morphology to repair the binary image.
        # This is the front ground image.
        self.imgFrontGround = self.getFrontGround(imgInput.copy())
        self.imgFrontGround = self.getClosing(self.imgFrontGround, iterations_erode=3, iterations_dilate=3, kernelSize=3)

        # Convert the input to HSV, use inRange to filter the HSV image, and apply morphology to fix repair the binary image
        # This is the sprout image.
        lower_hsv = np.array([27, 0,147], dtype=np.uint8)
        upper_hsv = np.array([180, 255, 255], dtype=np.uint8)
        self.imgSprout = self.getSproutImg(imgInput.copy(), lower_hsv, upper_hsv)
        self.imgSprout = self.getClosing(self.imgSprout, iterations_erode=3, iterations_dilate=3, kernelSize=3)

        # Add the front ground image and the sprout image in order to get the SeedAndSprout image.
        # This is the SeedAndSprout image
        self.imgSeedAndSprout = self.getSeedAndSproutImg(self.imgFrontGround, self.imgSprout)

    def getFrontGround(self, imgRGB):
        #Do the grayscale converting
        img_gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

        # Do the thresholding of the image.
        thresholdLevel = 128
        maxValue = 128
        ret, img_binary = cv2.threshold(img_gray, thresholdLevel, maxValue, cv2.THRESH_BINARY)
        return img_binary

    def getSproutImg(self, imgRGB, lower_hsv, upper_hsv):
        img_hsv = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
        img_sprout = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        return img_sprout

    def getSeedAndSproutImg(self, img_front_ground, img_sprout):
        img_seed_and_sprout = cv2.add(img_front_ground, img_sprout)
        return img_seed_and_sprout

    def getOpening(self, img_binary, iterations_erode, iterations_dilate, kernelSize):
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        img_erode = cv2.erode(img_binary, kernel, iterations=iterations_erode)
        img_morph = cv2.dilate(img_erode, kernel, iterations=iterations_dilate)

        # After the morph, the image will be shifter towards the origo with the
        # size of the kernel.
        # In order account for this, we do the following:
        #   Crop the bottom and right side of the morphed image
        #   equal to kernelsize - 1.
        #   Append zeropadding to top and left side equal to kernelsize - 1.
        height =  img_morph.shape[0]
        width =  img_morph.shape[1]
        crop = img_morph[0:height-kernelSize+1, 0:width-kernelSize+1]
        img_morph = cv2.copyMakeBorder(crop, kernelSize-1, 0, kernelSize-1, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_morph

    def getClosing(self, img_binary, iterations_erode, iterations_dilate, kernelSize):
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        img_dilate = cv2.dilate(img_binary, kernel, iterations=iterations_dilate)
        img_morph = cv2.erode(img_dilate, kernel, iterations=iterations_erode)

        # After the morph, the image will be shifter towards the origo with the
        # size of the kernel.
        # In order account for this, we do the following:
        #   Crop the bottom and right side of the morphed image
        #   equal to kernelsize - 1.
        #   Append zeropadding to top and left side equal to kernelsize - 1.
        height =  img_morph.shape[0]
        width =  img_morph.shape[1]
        crop = img_morph[0:height-kernelSize+1, 0:width-kernelSize+1]
        img_morph = cv2.copyMakeBorder(crop, kernelSize-1, 0, kernelSize-1, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_morph







