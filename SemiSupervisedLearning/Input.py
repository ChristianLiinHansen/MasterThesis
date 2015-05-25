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

    ##################################
    # Functions                      #
    ##################################

    def getTrainingDataImages(self):
        # OpenCV
        # imgTrainingData1 = cv2.imread(readFromImagePath + "class2TrainingCropped1.png", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTrainingData2 = cv2.imread(readFromImagePath + "class2TrainingCropped2.png", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTrainingData3 = cv2.imread(readFromImagePath + "class3.png", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTestData       = cv2.imread(readFromImagePath + "class0.png", cv2.CV_LOAD_IMAGE_COLOR)

        # PIL - Python Image Library
        imgTrainingData1 = Image.open(self.readFromImagePath + "class2.png")
        imgTrainingData2 = Image.open(self.readFromImagePath + "class2Training2.png")
        # imgTrainingData1 = Image.open(self.readFromImagePath + "class2TrainingCropped1.png")
        # imgTrainingData2 = Image.open(self.readFromImagePath + "class2TrainingCropped2.png")
        imgTrainingData3 = Image.open(self.readFromImagePath + "class3.png")
        imgTestData = Image.open(self.readFromImagePath + "class0.png")
        return imgTrainingData1, imgTrainingData2, imgTrainingData3, imgTestData

    def closeDown(self):
        cv2.destroyAllWindows()