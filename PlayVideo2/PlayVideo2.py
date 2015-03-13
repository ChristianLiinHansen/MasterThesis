#!/usr/bin/env python

import numpy as np
import cv2
import os

# Setting the FPS with cv and cv2 do not work with the camera Logitech e930.
# So the class CameraDriverCV was deleted, since the FPS did not work here. The cv2 is better, and fitted to the cv2.findContours etc...
# Last resort is to try inserting into a camera buffer, e.i have a buffer with size of 30.
# When the buffer is full, we take the last image and use to process.
# Then we empty the image buffer and do this over and over again...
# However this must be multithreading, otherwise we dont get rid of the delay.

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
        self.absoluteFocus = np.array(40, dtype=np.uint8)

         # sharpness (int)    : min=0 max=255 step=1 default=128 value=128
        self.sharpness = np.array(200, dtype=np.uint8)

        # Exposure min=3 max=2047 step=1 default=250 value=250 flags=inactive
        self.absoluteExposure = np.array(260, dtype=np.uint16)

        # Take a picture bottom
        self.bottom = np.array(0, dtype=np.uint8)

        # Active settings bottom
        self.adjustingSettings = np.array(0, dtype=np.uint8)

        # Horizontal cropping lines
        self.horizontalLines = np.array(200, dtype=np.uint16)

        # Vertical cropping lines
        self.verticalLines = np.array(500, dtype=np.uint16)

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

    def setSettings(self, absolutFocus, sharpness, absoluteExposure):
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

    #

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

    def absolutFocusTrackBar(self, nameOfWindow):
        self.absoluteFocus = self.trackbarListener("Absolute focus", nameOfWindow)

    def sharpnessTrackBar(self, nameOfWindow):
        self.sharpness = self.trackbarListener("Sharpness", nameOfWindow)

    def absoluteExposureTrackBar(self, nameOfWindow):
        self.absoluteExposure = self.trackbarListener("Absolute exposure", nameOfWindow)

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

def main():

    # "GUI Bottoms"
    adjustSettings = False
    pictureTaken = False

    # The image_show_ratio is to display images properly on this PC screen.
    image_show_ratio = 0.5

    # Calling the CameraDriver with cameraIndex as argument. Could switch to 1 og 2 sometimes...
    cd = CameraDriver(1)

    # Setting the names of different windows
    nameOfTrackBarWindow = "Trackbar settings"
    nameOfVideoStreamWindow = "VideoStream"
    nameOfVideoStreamWindowCropped = "VideoStream cropped"

    # Add the trackbar in the trackbar window
    cd.addTrackbar("Adjusting settings", nameOfTrackBarWindow, cd.adjustingSettings, 1)
    cd.addTrackbar("Absolute focus", nameOfTrackBarWindow, cd.absoluteFocus, 255)
    cd.addTrackbar("Sharpness", nameOfTrackBarWindow, cd.sharpness, 255)
    cd.addTrackbar("Absolute exposure", nameOfTrackBarWindow, cd.absoluteExposure, 2047)
    cd.addTrackbar("Horizontal crop", nameOfTrackBarWindow, cd.horizontalLines, int(cd.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)/6))
    cd.addTrackbar("Vertical crop", nameOfTrackBarWindow, cd.verticalLines, int(cd.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)/2))

    # Let the program run, until the user push ECS
    while True:

        # Listen for changes for adjusting the settings
        cd.adjustingSettingsTrackBar(nameOfTrackBarWindow)


        # Get an image from the camera
        image = cd.getImg()

        # If the adjustingSetting bottom is ON, then we accept a litte more delay until we are saticefied with the settings.
        if cd.adjustingSettings:

            # The trackbar setting update
            cd.horizontalLineTrackBar(nameOfTrackBarWindow)
            cd.verticallLineTrackBar(nameOfTrackBarWindow)
            cd.absolutFocusTrackBar(nameOfTrackBarWindow)
            cd.absoluteExposureTrackBar(nameOfTrackBarWindow)
            cd.sharpnessTrackBar(nameOfTrackBarWindow)
            cd.setSettings(cd.absoluteFocus, cd.sharpness, cd.absoluteExposure)
            imgDraw = cd.drawCroppingLines(image)
            croppedImg = cd.getCroppedImg(image, cd.verticalLines, cd.horizontalLines)
            cd.showImg(nameOfVideoStreamWindow, imgDraw, image_show_ratio)
            cd.showImg(nameOfVideoStreamWindowCropped, croppedImg, image_show_ratio)
            adjustSettings = True

        # Else the adjustingSetting bottom is OFF.
        else:

            # If we have been adjusting the settings previously, then we show the adjusted result
            if adjustSettings is True:

                # If this is the first time we se the result after adjusting the settings, we grap a picture of it for late use.
                if pictureTaken is False:
                    cd.saveImg("ImageCropped", croppedImg)
                    pictureTaken = True

                # Otherwise we just show the croped/ROI streaming video that has been adjusted with focus and ROI
                croppedImg = cd.getCroppedImg(image, cd.verticalLines, cd.horizontalLines)
                cd.showImg(nameOfVideoStreamWindowCropped, croppedImg, image_show_ratio)
                cv2.destroyWindow(nameOfVideoStreamWindow)

            # Else then we have not yet adjusted the settings...
            else:
                cd.showImg(nameOfVideoStreamWindow, image, image_show_ratio)

        # If the user push "ESC" the program ececutes
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cd.closeDown()
            break

if __name__ == '__main__':
    main()
