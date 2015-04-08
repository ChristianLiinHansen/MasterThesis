#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2

class Segmentation(object):

    def __init__(self, imgRGB, imgFrontGround, imgSeedAndSprout, classStamp):
        # Debug: Get the RGB input image over in this class to draw contours on the real image etc.
        self.imgRGB = imgRGB

        # Store which class the image is stamped
        self.classStamp = classStamp

        # Find the contours in the binary image
        self.contourFrontGround = self.getContours(imgFrontGround)

        # Filter out the number of contours, like small noise-blobs, etc.
        self.contoursFrontGroundFiltered, listOfAreas = self.getContoursFilter(self.contourFrontGround, 200, 2000)

        # Debugging. Draw the contours and store it in the imgContours.
        self.imgContours = self.drawContour(imgFrontGround, self.contoursFrontGroundFiltered, lineWidth=2)

        # Before doing any feature extraction, it is important to run through all the pixels, that is in the ROI
        # and not the edge pixels.
        self.getROI(self.contoursFrontGroundFiltered, imgSeedAndSprout, classStamp)

        # Do some feature extraction
        # self.features = self.getFeaturesFromContours(imgSeedAndSprout, self.contoursFrontGroundFiltered, self.classStamp)

    def getROI(self, contours, imgSeedAndSprout, classStamp):
        # Run through all the contours
        # indexCounter = 0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            p1 = (x, y)
            p2 = (x+width, y)
            p3 = (x+width, y+height)
            p4 = (x, y+height)

            # Debugging: Drawing the boundingBox for each contour.
            self.drawBoundingBox(p1, p2, p3, p4, self.imgContours, (255, 0, 0), 1)
            # cv2.imshow("Cropped ROI "+str(indexCounter), imgBBcropped)
            # indexCounter = indexCounter + 1

            # Crop out each boundingbox
            imgBBcropped = imgSeedAndSprout[y:y+height, x:x+width]

            # Run through all pixel for each contour. In order to remember where each contour is in respect to the
            # image with seed and sprout, each contour upper-left coordinate is remembered.
            rowCounter = y
            colCounter = x

            # List with sprout pixels
            sprout = []

            for row in imgBBcropped:
                for pixel in row:
                    if pixel == 255:
                        # print "Hey we found a white pixel at this location: (", colCounter, ",", rowCounter, ")"
                        pixelCoordinate = (colCounter, rowCounter)
                        sprout.append(pixelCoordinate)
                    colCounter = colCounter + 1
                rowCounter = rowCounter + 1
                colCounter = x

            if sprout:
                # Then convert it, in order to let it be used with the minAreaRect function
                sproutConvertedFormat = self.convertFormatForMinRectArea(sprout)

                # Then find the oriented bounding box of the sprouts
                obbSprout = cv2.minAreaRect(sproutConvertedFormat)

                # Then make sure that the "length" is the longest side of the boundingbox
                # and the "width" is the shortest side of the boundingbox.
                length, width = self.getLengthAndWidthFromSprout(obbSprout)
                p1, p2, p3, p4 = self.getBoxPoints(obbSprout)

                # Debug: Convert the imgSeedAndSproutImage to an color image, in order to draw color on it
                imgDraw = imgSeedAndSprout.copy()
                imgDraw = cv2.cvtColor(imgDraw, cv2.COLOR_GRAY2BGR)
                self.drawBoundingBox(p1, p2, p3, p4, imgDraw, (0, 0, 255), 1)

                # cv2.drawContours(mask, contours, -1, contourColor, lineWidth)
                cv2.imshow("Test of imgContours"+str(classStamp), imgDraw)
                cv2.imwrite("/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/imgDraw.png",imgDraw)
                cv2.waitKey(0)

            else:
                print "Hey this contour contained no sprout pixels from classstamp", classStamp

    def getBoxPoints(self, rect):
        # http://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        # p1 = (box[0][0], box[0][1])
        # p2 = (box[1][0], box[1][1])
        # p3 = (box[2][0], box[2][1])
        # p4 = (box[3][0], box[3][1])
        p1 = (box[0][1], box[0][0])
        p2 = (box[1][1], box[1][0])
        p3 = (box[2][1], box[2][0])
        p4 = (box[3][1], box[3][0])
        return p1, p2, p3, p4

    def getLengthAndWidthFromSprout(self, obb):
        # Get the length, width and angle
        tempLength = obb[1][0]
        tempWidth = obb[1][1]

        # Okay, so we know that all rectangles, the length is defined to be to longest axis and the width is the shortest.
        # So if the output of the length is shorter then the width, then we swop the two variables, so the length is the longest.

        if tempLength < tempWidth:
            length = tempWidth
            width = tempLength
        else:
            length = tempLength
            width = tempWidth
        return length, width

    def convertFormatForMinRectArea(self, listOfPixels):
        # print "The list of pixels within the convertFormatForMinRectArea is:", listOfPixels
        list = []
        for element in listOfPixels:
            x = element[0]
            y = element[1]
            stucture = [[y, x]]
            list.append(stucture)

        # Convert it to numpy
        list_np = np.array(list, dtype=np.int32)
        return list_np

    def getContours(string, imgFrontGround):
        #Copy the image, to avoid manipulating with original
        imgFrontGroundCopy = imgFrontGround.copy()

        #Find the contours of the thresholded image
        #Note: See OpenCV doc if needed to change the arguments in findContours.
        # Note: The CHAIN_APPROX_SIMPLE only takes the outer coordinates. I.e. a square has only four coordinates instead
        # of following the edge all around. The CHAIN_APPROX_NONE takes all the pixels around the edge
        contours, hierarchy = cv2.findContours(imgFrontGroundCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        #Return the contours. We dont want to use the hierarchy yet
        # However the hierarchy is usefull the detect contours inside a contour or dont detect it.
        # That is what hierarchy keeps track of. (Children and parents)
        return contours

    def getContoursFilter(self, contours, minAreaThreshold, maxAreaThreshold):
        temp_contour = []
        temp_contourArea = []
        contourAreaMax = 0
        contourAreaMin = maxAreaThreshold

        for contour in contours:
            #Get the area of the given contour, in order to check if that contour is actually something useful, like a seed or sprout.
            contour_area = cv2.contourArea(contour, False)

            if contour_area > contourAreaMax:
                contourAreaMax = contour_area

            if (contour_area < contourAreaMin):
                contourAreaMin = contour_area

            # If the area is below a given threshold, we skip that contour. It simply had to few pixels to represent an object = seed + sprout
            if (contour_area < minAreaThreshold) or (contour_area > maxAreaThreshold):
                # print "The contour area is", contour_area, "and hence skipped"
                continue

            temp_contourArea.append(contour_area)
            temp_contour.append(contour)

        # print "Now contours looks like this:", temp_contour
        # print "The contourAreaMax was:", contourAreaMax
        # print "The contourAreaMin was:", contourAreaMin
        return temp_contour, temp_contourArea

    def drawContour(self, img, contours, lineWidth):

        # Copy the image from the argument "img"
        img_copy = img.copy()

        # Create an empty image with same size as the input img.
        mask = np.zeros(img_copy.shape, dtype="uint8")

        # Convert the empty image to color map, in order to draw color on the image
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Draw the contours in the empty image mask
        contourColor = (0, 255, 0)
        cv2.drawContours(mask, contours, -1, contourColor, lineWidth)
        return mask

    def drawCentroid(self, image, centers, size, RGB_list):
        # Color the central coordinates for seeds with a filled circle
        for center in centers:
            cv2.circle(image, center, size, RGB_list, -1)

    def drawBoundingBox(self, p1, p2, p3, p4, imgDraw, boundingBoxColor, lineWidth):
        # Draw the oriente bouningbox
        lineWidth = 2
        cv2.line(imgDraw, p1, p2, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p2, p3, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p3, p4, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p4, p1, boundingBoxColor, lineWidth)
