#!/usr/bin/env python

import numpy as np
import cv2
import os
import threading
import Queue

# Setting the FPS with cv and cv2 do not work with the camera Logitech e930.
# So the class CameraDriverCV was deleted, since the FPS did not work here. The cv2 is better, and fitted to the cv2.findContours etc...
# Last resort is to try inserting into a camera buffer, e.i have a buffer with size of 30.
# When the buffer is full, we take the last image and use to process.
# Then we empty the image buffer and do this over and over again...
# However this must be multithreading, otherwise we dont get rid of the delay.

class ProducerImg:
    def __init__(self, cameraIndex):
        self.cameraIndex = cameraIndex
        self.cap = cv2.VideoCapture(cameraIndex)

        # Set the resolution
        self.setResolution()

        # Disable autofocus to begin with
        self.autoFocus = np.array(0, dtype=np.uint8)

        # Set focus to a specific value. High values for nearby objects and
        # low values for distant objects.
        self.absoluteFocus = np.array(0, dtype=np.uint8)

         # sharpness (int)    : min=0 max=255 step=1 default=128 value=128
        self.sharpness = np.array(0, dtype=np.uint8)

        # Take a picture bottom
        self.bottom = np.array(0, dtype=np.uint8)

        # Active settings bottom
        self.adjustingSettings = np.array(0, dtype=np.uint8)

        # Horizontal cropping lines
        self.horizontalLines = np.array(0, dtype=np.uint8)

        # Vertical cropping lines
        self.verticalLines = np.array(0, dtype=np.uint8)

    def drawCroppingLines(self, img):
        # Make a copy of the image
        imgDraw = img.copy()

        # Get the shape of the input image
        height = imgDraw.shape[0]
        width = imgDraw.shape[1]

        # Drawing the horizontal lines
        p1 = (0, self.horizontalLines)
        p2 = (width, self.horizontalLines)
        p3 = (0, height-self.horizontalLines)
        p4 = (width, height-self.horizontalLines)
        cv2.line(imgDraw, p1, p2, (0, 255, 0), 5)
        cv2.line(imgDraw, p3, p4, (0, 255, 0), 5)

        # Drawing the vertical lines
        p5 = (self.verticalLines, 0)
        p6 = (self.verticalLines, height)
        p7 = (width-self.verticalLines, 0)
        p8 = (width-self.verticalLines, height)
        cv2.line(imgDraw, p5, p6, (0, 255, 0), 5)
        cv2.line(imgDraw, p7, p8, (0, 255, 0), 5)
        return imgDraw

    def setResolution(self):
        # Set the camera in 1080p resolution. So the height is 1080, p = progressive scan = not interlaced. = "Full HD" = 1920 x 1080.
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)      # Slight delay with full HD in app. 1 sec.
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        print "Pixel width is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        print "Pixel height is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    def setAutoFocus(self, autoFocus, absolutFocus, sharpness):

        # If the autoFocus is ON, we use the autofocus
        if autoFocus:
            # os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=' + str(int(autoFocus)))
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=1')

        # Else we set the autoFocus to OFF, and use manuel focus
        else:
            # os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=' + str(int(autoFocus)))
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=0')
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_absolute=' + str(absolutFocus))
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c sharpness=' + str(sharpness))

    def getImg(self):
        if self.cap.isOpened():
            ret, img = self.cap.read()
            return img
        else:
            print 'Cant open video at cameraindex:', self.cameraIndex

    def showImg(self, nameOfWindow, image, scale):
        imgCopy = image.copy()
        image_show = self.scaleImg(imgCopy, scale)
        cv2.imshow(nameOfWindow, image_show)

    def scaleImg(self, image, scale):
        img_scale = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return img_scale

    def autoFocusTrackBar(self, nameOfWindow):
        self.autoFocus = self.trackbarListener("Autofocus", nameOfWindow)

    def absolutFocusTrackBar(self, nameOfWindow):
        self.absoluteFocus = self.trackbarListener("Absolute focus", nameOfWindow)

    def sharpnessTrackBar(self, nameOfWindow):
        self.sharpness = self.trackbarListener("Sharpness", nameOfWindow)

    def adjustingSettingsTrackBar(self, nameOfWindow):
        self.adjustingSettings = self.trackbarListener("Adjusting settings", nameOfWindow)

    def horizontalLineTrackBar(self, nameOfWindow):
        self.horizontalLines = self.trackbarListener("Horizontal crop", nameOfWindow)

    def verticallLineTrackBar(self, nameOfWindow):
        self.verticalLines = self.trackbarListener("Vertical crop", nameOfWindow)

    def trackbarListener(self, nameOfTrackbar, nameOfWindow):
        value = cv2.getTrackbarPos(nameOfTrackbar, nameOfWindow)
        return value

    def addTrackbar(self, nameOfTrackbar, nameOfWindow, value, maxValue):
        cv2.namedWindow(nameOfWindow)
        cv2.createTrackbar(nameOfTrackbar, nameOfWindow, value, maxValue, self.nothing)

    def nothing(self, x):
        pass

    def takePictureTrackBar(self, nameOfWindow):
        self.bottom = self.trackbarListener("Take picture", nameOfWindow)
        self.addTrackbar("Take picture", nameOfWindow, self.bottom, 1)

    def getCroppedImg(self, img, offset_x, offset_y):
        croppedImg = img.copy()
        croppedImg = croppedImg[offset_y:img.shape[0]-offset_y, offset_x:img.shape[1]-offset_x]
        # cv2.imshow(nameOfWindow, croppedImg)
        return croppedImg

    def saveImg(self, nameOfWindow, image):
        cv2.imwrite("/home/christian/workspace_python/MasterThesis/SeedDetection/writefiles/" + str(nameOfWindow) + ".jpg", image)

    def closeDown(self):
        print("User closed the program...")
        cv2.destroyAllWindows()
        self.cap.release()

class Consumer:
    print "Comsumer"

class CameraDriver:
    def __init__(self, cameraIndex):

        self.cameraIndex = cameraIndex
        self.cap = cv2.VideoCapture(cameraIndex)

        # Set the resolution
        self.setResolution()

        # Disable autofocus to begin with
        self.autoFocus = np.array(0, dtype=np.uint8)

        # Set focus to a specific value. High values for nearby objects and
        # low values for distant objects.
        self.absoluteFocus = np.array(0, dtype=np.uint8)

         # sharpness (int)    : min=0 max=255 step=1 default=128 value=128
        self.sharpness = np.array(128, dtype=np.uint8)

        # Take a picture bottom
        self.bottom = np.array(0, dtype=np.uint8)

    def setResolution(self):
        # Set the camera in 1080p resolution. So the height is 1080, p = progressive scan = not interlaced. = "Full HD" = 1920 x 1080.
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)      # Slight delay with full HD in app. 1 sec.
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)      # Better with delay
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 576)
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 848)
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

        print "Pixel width is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        print "Pixel height is:", self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    def setAutoFocus(self, autoFocus, absolutFocus, sharpness):

        # If the autoFocus is ON, we use the autofocus
        if autoFocus:
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=' + str(int(autoFocus)))

        # Else we set the autoFocus to OFF, and use manuel focus
        else:
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_auto=' + str(int(autoFocus)))
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c focus_absolute=' + str(absolutFocus))
            os.system('v4l2-ctl -d ' + str(self.cameraIndex) + ' -c sharpness=' + str(sharpness))

    def getImg(self):
        if self.cap.isOpened():
            ret, img = self.cap.read()
            return img
        else:
            print 'Cant open video at cameraindex:', self.cameraIndex

    def showImg(self, nameOfWindow, image, scale):
        imgCopy = image.copy()
        image_show = self.scaleImg(imgCopy, scale)
        cv2.imshow(nameOfWindow, image_show)

    def saveImg(self, nameOfWindow, image):
        cv2.imwrite("/home/christian/workspace_python/MasterThesis/SeedDetection/writefiles/" + str(nameOfWindow) + ".jpg", image)

    def scaleImg(self, image, scale):
        img_scale = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return img_scale

    def autoFocusTrackBar(self, nameOfWindow):
        self.autoFocus = self.trackbarListener("Autofocus", nameOfWindow)
        self.addTrackbar("Autofocus", nameOfWindow, self.autoFocus, 1)
        self.setAutoFocus(self.autoFocus, self.absoluteFocus, self.sharpness)

    def absolutFocusTrackBar(self, nameOfWindow):
        self.absoluteFocus = self.trackbarListener("Absolute focus", nameOfWindow)
        self.addTrackbar("Absolute focus", nameOfWindow, self.absoluteFocus, 255)

    def sharpnessTrackBar(self, nameOfWindow):
        self.sharpness = self.trackbarListener("Sharpness", nameOfWindow)
        self.addTrackbar("Sharpness", nameOfWindow, self.sharpness, 255)

    def takePictureTrackBar(self, nameOfWindow):
        self.bottom = self.trackbarListener("Take picture", nameOfWindow)
        self.addTrackbar("Take picture", nameOfWindow, self.bottom, 1)

    def addTrackbar(self, nameOfTrackbar, nameOfWindow, value, maxValue):
        cv2.namedWindow(nameOfWindow)
        cv2.createTrackbar(nameOfTrackbar, nameOfWindow, value, maxValue, self.nothing)

    def trackbarListener(self, nameOfTrackbar, nameOfWindow):
        value = cv2.getTrackbarPos(nameOfTrackbar, nameOfWindow)
        return value

    def nothing(self, x):
        pass

    def getCroppedImg(self, nameOfWindow, img):
        offset_x = 400
        offset_y = 200
        croppedImg = img.copy()
        croppedImg = croppedImg[offset_y:img.shape[0]-offset_y, offset_x:img.shape[1]-offset_x]
        cv2.imshow(nameOfWindow, croppedImg)
        return croppedImg

    def closeDown(self):
        print("User closed the program...")
        cv2.destroyAllWindows()
        self.cap.release()

def main():

    # q = Queue.Queue()
    # p = Producer()

    adjustSettings = False
    pictureTaken = False

    image_show_ratio = 0.5
    cd = ProducerImg(0)

    nameOfTrackBarWindow = "Trackbar settings"
    nameOfVideoStreamWindow = "VideoStream"
    nameOfVideoStreamWindowCropped = "VideoStream cropped"

    # Add the trackbar in a window
    cd.addTrackbar("Adjusting settings", nameOfTrackBarWindow, cd.adjustingSettings, 1)
    cd.addTrackbar("Autofocus", nameOfTrackBarWindow, cd.autoFocus, 1)
    cd.addTrackbar("Absolute focus", nameOfTrackBarWindow, cd.absoluteFocus, 255)
    cd.addTrackbar("Sharpness", nameOfTrackBarWindow, cd.sharpness, 255)
    cd.addTrackbar("Horizontal crop", nameOfTrackBarWindow, cd.horizontalLines, int(cd.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)/6))
    cd.addTrackbar("Vertical crop", nameOfTrackBarWindow, cd.verticalLines, int(cd.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)/2))

    while True:

        # Listen for changes on the trackbars
        cd.adjustingSettingsTrackBar(nameOfTrackBarWindow)
        cd.horizontalLineTrackBar(nameOfTrackBarWindow)
        cd.verticallLineTrackBar(nameOfTrackBarWindow)

        # Get an image from the camera
        image = cd.getImg()

        # If the adjustingSetting bottom is ON, then we accept a litte more delay until we are saticefied with the settings.
        if cd.adjustingSettings:

            # The trackbar setting update
            cd.autoFocusTrackBar(nameOfTrackBarWindow)
            cd.absolutFocusTrackBar(nameOfTrackBarWindow)
            cd.sharpnessTrackBar(nameOfTrackBarWindow)
            cd.setAutoFocus(cd.autoFocus, cd.absoluteFocus, cd.sharpness)
            imgDraw = cd.drawCroppingLines(image)
            croppedImg = cd.getCroppedImg(image, cd.verticalLines, cd.horizontalLines)
            cd.showImg(nameOfVideoStreamWindow, imgDraw, image_show_ratio)
            cd.showImg(nameOfVideoStreamWindowCropped, croppedImg, image_show_ratio)
            adjustSettings = True

        else:

            if adjustSettings is True:

                if pictureTaken is False:
                    cd.saveImg("ImageCropped", croppedImg)
                    pictureTaken = True

                croppedImg = cd.getCroppedImg(image, cd.verticalLines, cd.horizontalLines)
                cd.showImg(nameOfVideoStreamWindowCropped, croppedImg, image_show_ratio)
                cv2.destroyWindow(nameOfVideoStreamWindow)

            else:
                cd.showImg(nameOfVideoStreamWindow, image, image_show_ratio)

        # If the user push "ESC" the program ececutes
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cd.closeDown()
            break

if __name__ == '__main__':
    main()
