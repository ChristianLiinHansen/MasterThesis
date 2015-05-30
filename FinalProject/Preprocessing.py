#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Input import Input as inputTest

HSVtrackBar = False

class Preprocessing(object):

    def __init__(self, imgInput, classStamp, saveImagePath):
        self.imgInput = imgInput
        self.classStamp = classStamp
        self.saveImagePath = saveImagePath

        #################################################
        # Front ground image
        #################################################
        # cv2.imshow("The input RGB image class"+str(self.classStamp), imgInput)
        # cv2.imwrite(self.saveImagePath + "imgInputClass"+str(self.classStamp)+".png", imgInput)

        # Convert the input image to binary image and use morphology to repair the binary image.
        # This is the front ground image.
        self.imgFrontGround = self.getFrontGround(imgInput.copy())
        # cv2.imshow("The imgFrontGround image before any morph class"+str(self.classStamp), self.imgFrontGround)
        # cv2.imwrite(self.saveImagePath + "imgFrontGroundBeforeMorphClass"+str(self.classStamp)+".png", self.imgFrontGround)

        self.imgFrontGround = self.getClosing(self.imgFrontGround, iterations_erode=1, iterations_dilate=1, kernelSize=3, kernelShape=0)
        # cv2.imshow("The imgFrontGround image after 1 x closing class"+str(self.classStamp), self.imgFrontGround)
        # cv2.imwrite(self.saveImagePath + "imgFrontGroundAfter1xClosingClass"+str(self.classStamp)+".png", self.imgFrontGround)

        #################################################
        # Sprout image
        #################################################
        # Hue min and max
        self.hueMin = np.array(0, dtype=np.uint8)
        self.hueMax = np.array(180, dtype=np.uint8)
        # Saturation min and max
        self.saturationMin = np.array(0, dtype=np.uint8)
        self.saturationMax = np.array(118, dtype=np.uint8)
        # Value min and max
        self.valueMin = np.array(150, dtype=np.uint8)
        self.valueMax = np.array(255, dtype=np.uint8)

        # Convert the input to HSV, use inRange to filter the HSV image, and apply morphology to fix repair the binary image
        if HSVtrackBar:
            nameOfTrackBarWindow = "See the HSV result"
            self.addTrackbar("Hue min", nameOfTrackBarWindow, self.hueMin, 180)
            self.addTrackbar("Hue max", nameOfTrackBarWindow, self.hueMax, 180)
            self.addTrackbar("Saturation min", nameOfTrackBarWindow, self.saturationMin, 255)
            self.addTrackbar("Saturation max", nameOfTrackBarWindow, self.saturationMax, 255)
            self.addTrackbar("Value min", nameOfTrackBarWindow, self.valueMin, 255)
            self.addTrackbar("Value max", nameOfTrackBarWindow, self.valueMax, 255)

            while(1):
                self.hueMinTrackBar("Hue min", nameOfTrackBarWindow)
                self.hueMaxTrackBar("Hue max", nameOfTrackBarWindow)
                self.saturationMinTrackBar("Saturation min", nameOfTrackBarWindow)
                self.saturationMaxTrackBar("Saturation max", nameOfTrackBarWindow)
                self.valueMinTrackBar("Value min", nameOfTrackBarWindow)
                self.valueMaxTrackBar("Value max", nameOfTrackBarWindow)

                lower_hsv = np.array([self.hueMin, self.saturationMin, self.valueMin], dtype=np.uint8)
                upper_hsv = np.array([self.hueMax, self.saturationMax, self.valueMax], dtype=np.uint8)

                self.imgSprout = self.getSproutImg(imgInput.copy(), lower_hsv, upper_hsv)

                img_seed_and_sprout = cv2.add(self.imgFrontGround, self.imgSprout)
                cv2.imshow(nameOfTrackBarWindow, img_seed_and_sprout)
                cv2.waitKey(1)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        else:
            lower_hsv = np.array([self.hueMin, self.saturationMin, self.valueMin], dtype=np.uint8)
            upper_hsv = np.array([self.hueMax, self.saturationMax, self.valueMax], dtype=np.uint8)
            self.imgSprout = self.getSproutImg(imgInput.copy(), lower_hsv, upper_hsv)
            # Show and write the imgSprout before morph
            # cv2.imshow("The imgSprout before morph class"+str(self.classStamp), self.imgSprout)
            # cv2.imwrite(self.saveImagePath + "imgSproutBeforeMorph"+str(self.classStamp)+".png", self.imgSprout)

        # Show and write the imgSeedAndSprout before morph
        # img_seed_and_sprout = cv2.add(self.imgFrontGround, self.imgSprout)
        # cv2.imshow("Seed and sprout image before morph class"+str(self.classStamp), img_seed_and_sprout)
        # cv2.imwrite(self.saveImagePath + "imgSeedandSproutBeforeMorphClass"+str(self.classStamp)+".png", img_seed_and_sprout)

        # Show and write the imgSprout after morph
        self.imgSprout = self.getOpening(self.imgSprout, iterations_erode=1, iterations_dilate=1, kernelSize=3, kernelShape=0)
        # cv2.imshow("The imgSprout after morph class"+str(self.classStamp), self.imgSprout)
        # cv2.imwrite(self.saveImagePath + "imgSproutAfterMorph"+str(self.classStamp)+".png", self.imgSprout)

        # Add the front ground image and the sprout image in order to get the SeedAndSprout image.
        # Show and write the imgSeedAndSprout after morph
        self.imgSeedAndSprout = cv2.add(self.imgFrontGround, self.imgSprout)
        # cv2.imwrite(saveImagePath + "imgSeedAndSproutBeforeAND.png", self.imgSeedAndSprout)
        # cv2.imshow("Seed and sprout image after morph class"+str(self.classStamp), self.imgSeedAndSprout)
        # cv2.imwrite(self.saveImagePath + "imgSeedandSproutAfter1xOpeningClass"+str(self.classStamp)+".png", self.imgSeedAndSprout)

        # Finally try to connect the blobs in the sprout area by dilate and use a AND operater
        # So take the sprout image and dilate in order to connect the broken sprout together, but
        # in order to not just flood out in the dark area, a AND operation can be performed between the result and the front ground image

        # if not classStamp == 1:
        kernel = np.ones((3, 3), np.uint8)
        kernel = np.matrix(([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), np.uint8)

        # cv2.imwrite(saveImagePath + "imgSprout.png", self.imgSprout)

        # First dilate to build the bridge
        imgSproutMorph = cv2.dilate(self.imgSprout, kernel, iterations=1)
        # Then erode after
        # imgSproutMorph = cv2.erode(imgSproutMorph, kernel, iterations=2)
        # cv2.imwrite(saveImagePath + "imgSproutMorph.png", imgSproutMorph)

        # Get a white front ground image, just by add two front ground image together. 128 + 128 = 255
        imgFrontGroundWhite = cv2.add(self.imgFrontGround, self.imgFrontGround)
        # cv2.imwrite(saveImagePath + "imgFrontGroundWhite.png", imgFrontGroundWhite)

        # Do a AND operation between the white front ground and the morphed
        self.imgSproutRepaired = cv2.bitwise_and(imgSproutMorph, imgFrontGroundWhite)
        # cv2.imwrite(saveImagePath + "imgSproutRepaired.png", self.imgSproutRepaired)

        # After this we add the new mask to a normal front ground image
        self.imgSeedandSproutRepaired = cv2.add(self.imgSproutRepaired, self.imgFrontGround)
        # cv2.imwrite(saveImagePath + "imgSeedandSproutRepaired.png", self.imgSeedandSproutRepaired)

            # And show the result
            # cv2.imshow("imgSeedandSproutRepairedClass"+str(classStamp), self.imgSeedandSproutRepaired)

    ##################################################################################
    # All the function for tracbars START... Could not found out to use them from Input.py #
    ##################################################################################
    def hueMinTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.hueMin = self.trackbarListener(nameOfTrackbar, nameOfWindow)
    def hueMaxTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.hueMax = self.trackbarListener(nameOfTrackbar, nameOfWindow)
    def saturationMinTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.saturationMin = self.trackbarListener(nameOfTrackbar, nameOfWindow)
    def saturationMaxTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.saturationMax = self.trackbarListener(nameOfTrackbar, nameOfWindow)
    def valueMinTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.valueMin = self.trackbarListener(nameOfTrackbar, nameOfWindow)
    def valueMaxTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.valueMax = self.trackbarListener(nameOfTrackbar, nameOfWindow)
    def trackbarListener(self, nameOfTrackbar, nameOfWindow):
        value = cv2.getTrackbarPos(nameOfTrackbar, nameOfWindow)
        return value
    def addTrackbar(self, nameOfTrackbar, nameOfWindow, value, maxValue):
        cv2.namedWindow(nameOfWindow)
        cv2.createTrackbar(nameOfTrackbar, nameOfWindow, value, maxValue, self.nothing)
    def nothing(self, x):
        pass

    ##################################################################################
    # All the function for tracbars END... Could not found out to use them from Input.py #
    ##################################################################################

    def SaveImages(self):
        cv2.imwrite(self.saveImagePath + "imgInputClass"+str(self.classStamp)+".png", self.imgInput)
        cv2.imwrite(self.saveImagePath + "imgFrontGroundClass"+str(self.classStamp)+".png", self.imgFrontGround)
        cv2.imwrite(self.saveImagePath + "imgSproutClass"+str(self.classStamp)+".png", self.imgSprout)

    def getFrontGround(self, imgRGB):
        #Do the grayscale converting
        img_gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Grayscale img", img_gray)

        # It seems that around 80-100 is a OK threshold
        # To find the optimal threshold, we use the THRESH_OTSU, which is based on the Otsu's optimal threshold algorithm

        # Do the thresholding of the image.
        thresholdLevel = 128 # This manual threshold is ignore, when the THRESH_OTSU is used. The "
        maxValue = 128
        OtsuOptimalThreshold, img_binary = cv2.threshold(img_gray, thresholdLevel, maxValue, cv2.THRESH_OTSU)
        # print "So Otso says the optimal threshold is:", OtsuOptimalThreshold

        # # Plot the histogram of the grayscale image in order to argue for choosen the threshold value
        # self.size = 18
        # font = {'size': self.size}
        # plt.rc('xtick', labelsize=self.size)
        # plt.rc('ytick', labelsize=self.size)
        # plt.rc('font', **font)
        #
        # self.fig = plt.figure(1, figsize=(10, 8.21), dpi=100, facecolor='w', edgecolor='k')
        # self.fig.suptitle("Histogram of grayscale image class"+str(self.classStamp), fontsize=22, fontweight='normal')
        # self.ax = self.fig.add_subplot(111)
        # self.fig.subplots_adjust(top=0.90)
        # # self.ax.set_title(self.lowerTitle)
        # plt.xlabel("Intensity bins", fontsize=self.size)
        # plt.ylabel("Frequency", fontsize=self.size)
        # plt.xlim(0, 255)
        # plt.ylim(0, 120000)
        # plt.grid(True)
        # plt.hist(img_gray.flatten(), 256)
        # self.ax.annotate("Otsu's global threshold = "+str(int(OtsuOptimalThreshold)), xy=(OtsuOptimalThreshold, 5000), xytext=(OtsuOptimalThreshold, 20000),
        #     arrowprops=dict(facecolor='black', shrink=0.001))
        # plt.savefig(self.saveImagePath + "HistogramOtsuClass"+str(self.classStamp)+".png")
        # plt.show()

        return img_binary

    def getSproutImg(self, imgRGB, lower_hsv, upper_hsv):
        # img_hsv = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2HSV)
        img_sprout = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        return img_sprout

    def getSeedAndSproutImg(self, img_front_ground, img_sprout):
        img_seed_and_sprout = cv2.add(img_front_ground, img_sprout)
        return img_seed_and_sprout

    def getOpening(self, img_binary, iterations_erode, iterations_dilate, kernelSize, kernelShape):
        if kernelShape == 0:
            kernel = np.ones((kernelSize, kernelSize), np.uint8)
        elif kernelShape == 1:
            kernel = np.matrix(([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), np.uint8)

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
        crop = img_morph[0:height-iterations_erode+1, 0:width-iterations_erode+1]
        img_morph = cv2.copyMakeBorder(crop, iterations_erode-1, 0, iterations_erode-1, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_morph

    def getClosing(self, img_binary, iterations_erode, iterations_dilate, kernelSize, kernelShape):
        if kernelShape == 0:
            kernel = np.ones((kernelSize, kernelSize), np.uint8)
        elif kernelShape == 1:
            kernel = np.matrix(([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), np.uint8)
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
        crop = img_morph[0:height-iterations_erode+1, 0:width-iterations_erode+1]
        img_morph = cv2.copyMakeBorder(crop, iterations_erode-1, 0, iterations_erode-1, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_morph







