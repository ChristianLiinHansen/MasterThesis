#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
import math
import random
class Segmentation(object):

    def __init__(self, saveToImagePath, readFromImagePath, imgMorph, imgRGB):
        self.saveToImagePath = saveToImagePath
        self.readFromImagePath = readFromImagePath

        # Find the contours in the binary image
        self.contoursImgMorph = self.getContours(imgMorph)

        # Filter out the number of contours, like small noise-blobs, etc.
        self.contoursFrontGroundFiltered, listOfAreas, listOfTooSmallContourAreas, listOfTooSmallContour = self.getContoursFilter(self.contoursImgMorph, 1, 4000)

        # Debugging. Draw the contours and store it in the imgContours.
        self.imgContours = self.drawContour(imgMorph, self.contoursFrontGroundFiltered, lineWidth= 2)

        # Calculate the center of mass (COM) for each contour
        centers = self.getCOMofAllContours(self.contoursFrontGroundFiltered)

        # Draw centers on a copy of the RGB input image
        self.imgDraw = self.drawCOMs(imgRGB, centers, 4, (255,0,0))


    def getHuMomentsOfAllContours(self, contours, huMomentIndex):
        hueMoments = []
        for contour in contours:
            temp = self.getHumomentOfSingleContour(contour, huMomentIndex)
            # temp = np.array(temp)   # Converting from list to array. Remember that temp return from getHumoment was a list with one element. To get acces i should do temp[0]
            if math.isnan(temp[0]):
                continue
            else:
                hueMoments.append(temp[0])
        return hueMoments

    def getHumomentOfSingleContour(self, contour, huMomentIndex):
        np_array = np.array(contour)
        #Calculate the moments for each contour in contours
        m = cv2.moments(np_array)
        Humoments = cv2.HuMoments(m)
        Humoments_after_log = Humoments
        # Humoments_after_log = -np.sign(Humoments[huMomentIndex])*np.log10(np.abs(Humoments[huMomentIndex]))
        return Humoments_after_log.tolist()

    def getCOMofAllContours(self, contours):
        centers = []
        for contour in contours:
            centers.append(self.getCOMofSingleContour(contour))
        return centers

    def getCOMofSingleContour(self, contour):
        np_array = np.array(contour)

        #Calculate the moments for each contour in contours
        m = cv2.moments(np_array)

        #If somehow one of the moments is zero, then we brake and reenter the loop (continue)
        #to avoid dividing with zero
        if (int(m['m01']) == 0 or int(m['m00'] == 0)):
            print "ops, devided by zero"
            # If we return None, then the program crash. So if there is an invalid division, then the COM is just 0.0
            return 0, 0

        #Calculate the centroid x,y, coordinate out from standard formula.
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        return center

    def getContours(string, imgMorph):
        #Copy the image, to avoid manipulating with original
        imgMorphCopy = imgMorph.copy()

        #Find the contours of the thresholded image
        #Note: See OpenCV doc if needed to change the arguments in findContours.
        # Note: The CHAIN_APPROX_SIMPLE only takes the outer coordinates. I.e. a square has only four coordinates instead
        # of following the edge all around. The CHAIN_APPROX_NONE takes all the pixels around the edge
        contours, hierarchy = cv2.findContours(imgMorphCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        #Return the contours. We dont want to use the hierarchy yet
        # However the hierarchy is usefull the detect contours inside a contour or dont detect it.
        # That is what hierarchy keeps track of. (Children and parents)
        return contours

    def getContoursFilter(self, contours, minAreaThreshold, maxAreaThreshold):
        temp_contour = []
        temp_contourArea = []
        tooSmallContourArea = []
        tooSmallContour = []
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
                tooSmallContourArea.append(contour_area)
                tooSmallContour.append(contour)
                continue

            temp_contourArea.append(contour_area)
            temp_contour.append(contour)

        return temp_contour, temp_contourArea, tooSmallContourArea, tooSmallContour

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

    def drawCOMs(self, image, centers, size, RGB_list):
        # Color the central coordinates for seeds with a filled circle
        imageDraw = image.copy()
        for center in centers:
            cv2.circle(imageDraw, center, size, RGB_list, -1)
        return imageDraw