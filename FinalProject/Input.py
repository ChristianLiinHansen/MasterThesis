#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
import os

class Input(object):
    def __init__(self, cameraIndex):
        self.cameraIndex = cameraIndex
        self.cap = cv2.VideoCapture(cameraIndex)
        self.cameraIsOpen = self.checkCamera()

        # Set the resolution
        if self.cameraIsOpen:
            self.setResolution()
            self.setV4L2()

    def checkCamera(self):
        if self.cap.isOpened():
            return True
        else:
            return False

    def setResolution(self):
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)

    def setV4L2(self):
        # Only works for hardware that support the following "video for linux 2" settings (v4l2).
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=0')
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_absolute=40')
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c sharpness=200')
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c exposure_auto=1')
        os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c exposure_absolute=260')

    def getImg(self):
        ret, img = self.cap.read()
        return img

    def closeDown(self):
        print("User closed the program...")
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