#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import cv2
import os
import numpy as np
from CameraDriver import CameraDriver

class Input(object):

    def __init__(self, cameraIndex):
        self.cameraIndex = cameraIndex
        self.cap = cv2.VideoCapture(cameraIndex)
        self.cameraIsOpen = self.checkCamera()

        # Activate system bottom
        self.buttonStartSystem = np.array(0, dtype=np.uint8)

        # Disable autofocus to begin with
        self.autoFocus = np.array(0, dtype=np.uint8)

        # Set focus to a specific value. High values for nearby objects and
        # low values for distant objects.
        self.absoluteFocus = np.array(40, dtype=np.uint8)

        # Exposure min=3 max=2047 step=1 default=250 value=250 flags=inactive
        self.absoluteExposure = np.array(260, dtype=np.uint16)

         # sharpness (int)    : min=0 max=255 step=1 default=128 value=128
        self.sharpness = np.array(200, dtype=np.uint8)

        # Horizontal cropping lines
        self.horizontalLines = np.array(100, dtype=np.uint16)

        # Vertical cropping lines
        self.verticalLines = np.array(420, dtype=np.uint16)

        # Initialize the training data
        # Doing the 2 classes classification
        # self.trainingData1 = self.getTrainingDataImages()[0]
        # self.trainingDataNeg1 = self.getTrainingDataImages()[1]
        # self.testingData = self.getTrainingDataImages()[2]

        # Doing the 3 classes classification
        self.trainingData1 = self.getTrainingDataImages()[0]
        self.trainingData2 = self.getTrainingDataImages()[1]
        self.trainingData3 = self.getTrainingDataImages()[2]
        self.testingData = self.getTrainingDataImages()[3]

        # Set the resolution
        if self.cameraIsOpen:
            self.setResolution()

    #########################################
    # Trackbar functions                    #
    ########################################

    def trackbarListener(self, nameOfTrackbar, nameOfWindow):
        value = cv2.getTrackbarPos(nameOfTrackbar, nameOfWindow)
        return value

    def addTrackbar(self, nameOfTrackbar, nameOfWindow, value, maxValue):
        cv2.namedWindow(nameOfWindow)
        cv2.createTrackbar(nameOfTrackbar, nameOfWindow, value, maxValue, self.nothing)

    def nothing(self, x):
        pass

    def startTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.buttonStartSystem = self.trackbarListener(nameOfTrackbar, nameOfWindow)

    def absoluteExposureTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.absoluteExposure = self.trackbarListener(nameOfTrackbar, nameOfWindow)

    def absoluteFocusTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.absoluteFocus = self.trackbarListener(nameOfTrackbar, nameOfWindow)

    def sharpnessTrackBar(self, nameOfTrackbar, nameOfWindow):
        self.sharpness = self.trackbarListener(nameOfTrackbar, nameOfWindow)


    #########################################
    # Other functions                      #
    ########################################

    def checkCamera(self):
        if self.cap.isOpened():
            return True
        else:
            return False

    def getTrainingDataImages(self):
        # imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTrainingClassNeg1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_lang_og_krum.jpg", cv2.CV_LOAD_IMAGE_COLOR)

        imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/3Classes/NGR_forkorteDEBUG.png", cv2.CV_LOAD_IMAGE_COLOR)
        imgTrainingClass2 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/3Classes/NGR_lang_og_krum.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        imgTrainingClass3 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/3Classes/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)


        # DEBUGGING!. NOTE: this function is only suppose to read two, or perhaps three training data images. Not a testing image.
        # This testing image, should come from the webcamera.
        # However in order to have some testing data, a still image is used, in order to verify the preprocessing, segmentation, classification and output component.
        # imgTestData = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_Mix.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTestData = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimaleDEBUG2.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTestData = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_lang_og_krumDEBUG2.jpg", cv2.CV_LOAD_IMAGE_COLOR)
        imgTestData = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/readfiles/ImageCropped.png", cv2.CV_LOAD_IMAGE_COLOR)
        # imgTestData = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section6/TestingInRoboLab/ImageCropped.png", cv2.CV_LOAD_IMAGE_COLOR)
        # return imgTrainingClass1, imgTrainingClassNeg1, imgTestData
        return imgTrainingClass1, imgTrainingClass2, imgTrainingClass3, imgTestData

    def setResolution(self):
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        print "Pixel width is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        print "Pixel height is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    def setV4L2(self, absolutFocus, absoluteExposure, sharpness):
        # Only works for hardware that support the following "video for linux 2" settings (v4l2).

        # Set autofocus OFF
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=0')

        # Set exposure to Manuel mode  # Choise of auto expusure see --> https://groups.google.com/forum/#!msg/plots-infrared/lSwIqQPJSY8/ZE-LcIj7V-wJ
        # exposure_auto (menu) : min=0 max=3 default=3    value=3  (0: Auto Mode 1: Manual Mode, 2: Shutter Priority Mode, 3: Aperture Priority Mode)
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c exposure_auto=1')

        # Set the absolute focus
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_absolute=' + str(absolutFocus))

        # Set the absolute exposure
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c exposure_absolute=' + str(absoluteExposure))

        # Set the sharpness
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c sharpness=' + str(sharpness))

        # os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_absolute=40')
        # os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c sharpness=200')
        # os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c exposure_auto=1')
        # os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c exposure_absolute=260')

    def getImg(self):
        ret, img = self.cap.read()
        return img

    def getCroppedImg(self):
        img = self.getImg()
        croppedImg = img.copy()
        croppedImg = croppedImg[self.horizontalLines:img.shape[0]-self.horizontalLines, self.verticalLines:img.shape[1]-self.verticalLines]
        return croppedImg

    def closeDown(self):
        cv2.destroyAllWindows()
        self.cap.release()

# class CameraDriver:
#     def __init__(self, cameraIndex):
#
#         self.cameraIndex = cameraIndex
#         self.cap = cv2.VideoCapture(cameraIndex)
#
#         # Set the resolution
#         self.setResolution()
#
#         # Disable autofocus to begin with
#         self.autoFocus = np.array(0, dtype=np.uint8)
#
#         # Set focus to a specific value. High values for nearby objects and
#         # low values for distant objects.
#         self.absoluteFocus = np.array(0, dtype=np.uint8)
#
#          # sharpness (int)    : min=0 max=255 step=1 default=128 value=128
#         self.sharpness = np.array(128, dtype=np.uint8)
#
#         # Take a picture bottom
#         self.bottom = np.array(0, dtype=np.uint8)
#
#     def setResolution(self):
#         # Set the camera in 1080p resolution. So the height is 1080, p = progressive scan = not interlaced. = "Full HD" = 1920 x 1080.
#         self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)      # Slight delay with full HD in app. 1 sec.
#         self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)      # Better with delay
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 576)
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 848)
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
#         # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
#
#         print "Pixel width is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
#         print "Pixel height is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
#
#     def setAutoFocus(self, autoFocus, absolutFocus, sharpness):
#
#         # If the autoFocus is ON, we use the autofocus
#         if autoFocus:
#             os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=' + str(int(autoFocus)))
#
#         # Else we set the autoFocus to OFF, and use manuel focus
#         else:
#             os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=' + str(int(autoFocus)))
#             os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_absolute=' + str(absolutFocus))
#             os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c sharpness=' + str(sharpness))
#
#     def getImg(self):
#         if self.cap.isOpened():
#             ret, img = self.cap.read()
#             return img
#         else:
#             print 'Cant open video at cameraindex:', self.cameraIndex
#
#     def showImg(self, nameOfWindow, image, scale):
#         imgCopy = image.copy()
#         image_show = self.scaleImg(imgCopy, scale)
#         cv2.imshow(nameOfWindow, image_show)
#
#     def saveImg(self, nameOfWindow, image):
#         cv2.imwrite("/home/christian/workspace_python/MasterThesis/SeedDetection/writefiles/" + str(nameOfWindow) + ".jpg", image)
#
#     def scaleImg(self, image, scale):
#         img_scale = cv2.resize(image, (0, 0), fx=scale, fy=scale)
#         return img_scale
#
#     def autoFocusTrackBar(self, nameOfWindow):
#         self.autoFocus = self.trackbarListener("Autofocus", nameOfWindow)
#         self.addTrackbar("Autofocus", nameOfWindow, self.autoFocus, 1)
#         self.setAutoFocus(self.autoFocus, self.absoluteFocus, self.sharpness)
#
#     def absolutFocusTrackBar(self, nameOfWindow):
#         self.absoluteFocus = self.trackbarListener("Absolute focus", nameOfWindow)
#         self.addTrackbar("Absolute focus", nameOfWindow, self.absoluteFocus, 255)
#
#     def sharpnessTrackBar(self, nameOfWindow):
#         self.sharpness = self.trackbarListener("Sharpness", nameOfWindow)
#         self.addTrackbar("Sharpness", nameOfWindow, self.sharpness, 255)
#
#     def takePictureTrackBar(self, nameOfWindow):
#         self.bottom = self.trackbarListener("Take picture", nameOfWindow)
#         self.addTrackbar("Take picture", nameOfWindow, self.bottom, 1)
#
#     def addTrackbar(self, nameOfTrackbar, nameOfWindow, value, maxValue):
#         cv2.namedWindow(nameOfWindow)
#         cv2.createTrackbar(nameOfTrackbar, nameOfWindow, value, maxValue, self.nothing)
#
#     def trackbarListener(self, nameOfTrackbar, nameOfWindow):
#         value = cv2.getTrackbarPos(nameOfTrackbar, nameOfWindow)
#         return value
#
#     def nothing(self, x):
#         pass
#
#     def getCroppedImg(self, nameOfWindow, img):
#         offset_x = 400
#         offset_y = 200
#         croppedImg = img.copy()
#         croppedImg = croppedImg[offset_y:img.shape[0]-offset_y, offset_x:img.shape[1]-offset_x]
#         cv2.imshow(nameOfWindow, croppedImg)
#         return croppedImg
#
#     def closeDown(self):
#         print("User closed the program...")
#         cv2.destroyAllWindows()
#         self.cap.release()