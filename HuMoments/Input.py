#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 24-5-2015
@author: Christian Liin Hansen
"""

import cv2
from PIL import Image

class Input(object):

    def __init__(self, saveToImagePath, readFromImagePath):
        self.saveToImagePath = saveToImagePath
        self.readFromImagePath = readFromImagePath

        self.trainingData1 = self.getTrainingDataImages()[0]
        self.trainingData2 = self.getTrainingDataImages()[1]
        self.trainingData3 = self.getTrainingDataImages()[2]
        self.testingData = self.getTrainingDataImages()[3]

    def getTrainingDataImages(self):
        # PIL - Python Image Library
        # imgTrainingData1 = Image.open(self.readFromImagePath + "class1.png")
        # imgTrainingData2 = Image.open(self.readFromImagePath + "class2.png")
        # imgTrainingData3 = Image.open(self.readFromImagePath + "class3.png")
        # testingData = Image.open(self.readFromImagePath + "class0.png")
        # OpenCV

        # Looking at the RGB image
        # imgTrainingData1 = cv2.imread(self.readFromImagePath + "class1.png", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTrainingData2 = cv2.imread(self.readFromImagePath + "class2.png", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTrainingData3 = cv2.imread(self.readFromImagePath + "class3.png", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTestData      = cv2.imread(self.readFromImagePath + "class0.png", cv2.CV_LOAD_IMAGE_COLOR)

        # Looking at the sprout images...
        imgTrainingData1 = cv2.imread(self.readFromImagePath + "sprout1.png", cv2.CV_LOAD_IMAGE_COLOR)
        imgTrainingData2 = cv2.imread(self.readFromImagePath + "sprout2.png", cv2.CV_LOAD_IMAGE_COLOR)
        imgTrainingData3 = cv2.imread(self.readFromImagePath + "sprout3.png", cv2.CV_LOAD_IMAGE_COLOR)
        imgTestData      = cv2.imread(self.readFromImagePath + "sprout0.png", cv2.CV_LOAD_IMAGE_COLOR)
        return imgTrainingData1, imgTrainingData2, imgTrainingData3, imgTestData

    def closeDown(self):
        cv2.destroyAllWindows()