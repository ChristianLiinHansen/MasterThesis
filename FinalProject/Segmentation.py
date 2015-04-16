#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
import random

class Segmentation(object):

    def __init__(self, imgRGB, imgFrontGround, imgSeedAndSprout, imgSprout, classStamp):
        # Debug: Get the RGB input image over in this class to draw contours on the real image etc.
        self.imgRGB = imgRGB

        # Store which class the image is stamped
        self.classStamp = classStamp

        # Find the contours in the binary image
        self.contoursFrontGround = self.getContours(imgFrontGround)

        # Filter out the number of contours, like small noise-blobs, etc.
        self.contoursFrontGroundFiltered, listOfAreas = self.getContoursFilter(self.contoursFrontGround, 200, 2000)

        # Debugging. Draw the contours and store it in the imgContours.
        self.imgContours = self.drawContour(imgFrontGround, self.contoursFrontGroundFiltered, lineWidth=2)

        # Before doing any feature extraction, it is important to run through all the pixels, that is in the ROI
        # and not the edge pixels.

        self.getROI(self.contoursFrontGroundFiltered, imgSeedAndSprout, imgSprout, classStamp)

        # Do some feature extraction
        # self.features = self.getFeaturesFromContours(imgSeedAndSprout, self.contoursFrontGroundFiltered, self.classStamp)

    def getLongestList(self, list1, list2):
        if len(list1) > len(list2):
            return list1
        else:
            return list2

    def runKmediansAlgorithm(self, sprout):

        print 20*'-',"Step 1", 20*'-'
        print "The sprout contains this:", sprout, " with a length of:", len(sprout)
        # Step 1 in K-means algorithm.
        # Randomly pick a center location from the sprout. In this case we only have K = 2

        # Make sure that the centerCluster1 are different from centerCluster2
        # centerCluster1 = (485, 502)
        # centerCluster2 = (486, 503)
        centerCluster1 = 0
        centerCluster2 = 0
        while centerCluster1 == centerCluster2:
            centerCluster1 = random.choice(sprout)
            centerCluster2 = random.choice(sprout)

        print "The first random coordinate is:", centerCluster1
        print "The secound random coordinate is:", centerCluster2

        iterationCounter = 0
        # Let K-medians runs at least 1 time
        while True:
            # Defines the cluster arrays
            cluster1List = []
            cluster2List = []
            iterationCounter = iterationCounter + 1

            print "\n", 20*'-',"Step 2", 20*'-'

            for element in sprout:
                # Calculate the distance from each element in sprout to the random picked pixel locations.

                d1 = np.sqrt(np.sum(np.power(np.array(centerCluster1)-np.array(element),2)))
                d2 = np.sqrt(np.sum(np.power(np.array(centerCluster2)-np.array(element),2)))

                # If the euclidian distance between the ranPixLoc1 and the pixellocation in the sprout list is less
                # than the euclidian distance between the ranPixLoc2 and the pixellocation in the sprout list,
                # than we assign the element to the cluster sproutCluster
                if d1 < d2:
                    # print "d1 < d2"
                    # print "cluster1List contains before:", cluster1List
                    cluster1List.append(element)
                    # print "cluster1List contains after:", cluster1List

                # Else d2 < d1 and we assign the element to cluster nonSproutCluster
                else:
                    # print "d2 < d1"
                    # print "cluster2List contains before:", cluster2List
                    cluster2List.append(element)
                    # print "cluster2List contains after:", cluster2List

            print "Now all the elements has run through. Cluster1List is:", cluster1List, " and length is:", len(cluster1List)
            print "Now all the elements has run through. cluster2List is:", cluster2List, " and length is:", len(cluster2List)

            # Step 3: Update the centers for each clusters
            cluster1ListMedian = tuple(map(np.median, zip(*cluster1List)))
            cluster1ListMedianRound = (int(round(cluster1ListMedian[0],0)), int(round(cluster1ListMedian[1],0)))
            newCenterCluster1 = cluster1ListMedianRound

            cluster2ListMedian = tuple(map(np.median, zip(*cluster2List)))
            cluster2ListMedianRound = (int(round(cluster2ListMedian[0],0)), int(round(cluster2ListMedian[1],0)))
            newCenterCluster2 = cluster2ListMedianRound

            print "The median newCenterCluster1 is:", newCenterCluster1
            print "The median newCenterCluster2 is:", newCenterCluster2

            # Check if we should break the algorithm
            if (centerCluster1 == newCenterCluster1) and (centerCluster2 == newCenterCluster2):
                print "Breaking the algorithm after the following iterations:", iterationCounter
                break
            else:
                # Update the center
                centerCluster1 = newCenterCluster1
                centerCluster2 = newCenterCluster2
        return cluster1List, cluster2List

    def runKmeansAlgorithm(self, sprout):

        # print 20*'-',"Step 1", 20*'-'
        # print "The sprout contains this:", sprout, " with a length of:", len(sprout)
        # Step 1 in K-means algorithm.
        # Randomly pick a center location from the sprout. In this case we only have K = 2

        # Make sure that the centerCluster1 are different from centerCluster2
        # Now with hardcodede start values. This is in order to check how many iterations it goes, when the points are close to each other.
        # centerCluster1 = (485, 502)
        # centerCluster2 = (486, 503)
        centerCluster1 = 0
        centerCluster2 = 0
        while centerCluster1 == centerCluster2:
            centerCluster1 = random.choice(sprout)
            centerCluster2 = random.choice(sprout)

        # print "The first random coordinate is:", centerCluster1
        # print "The secound random coordinate is:", centerCluster2

        iterationCounter = 0
        # Let K-means runs at least 1 time
        while True:
            # Defines the cluster arrays
            cluster1List = []
            cluster2List = []
            iterationCounter = iterationCounter + 1

            # print "\n", 20*'-',"Step 2", 20*'-'

            for element in sprout:
                # Calculate the distance from each element in sprout to the random picked pixel locations.

                d1 = np.sqrt(np.sum(np.power(np.array(centerCluster1)-np.array(element),2)))
                d2 = np.sqrt(np.sum(np.power(np.array(centerCluster2)-np.array(element),2)))

                # If the euclidian distance between the ranPixLoc1 and the pixellocation in the sprout list is less
                # than the euclidian distance between the ranPixLoc2 and the pixellocation in the sprout list,
                # than we assign the element to the cluster sproutCluster
                if d1 < d2:
                    cluster1List.append(element)

                # Else d2 < d1 and we assign the element to cluster nonSproutCluster
                else:
                    cluster2List.append(element)

            # print "Now all the elements has run through. Cluster1List is:", cluster1List, " and length is:", len(cluster1List)
            # print "Now all the elements has run through. cluster2List is:", cluster2List, " and length is:", len(cluster2List)

            # Step 3: Update the centers for each clusters
            cluster1ListMean = tuple(map(np.mean, zip(*cluster1List)))
            cluster1ListMeanRound = (int(round(cluster1ListMean[0],0)), int(round(cluster1ListMean[1],0)))
            newCenterCluster1 = cluster1ListMeanRound

            cluster2ListMean = tuple(map(np.mean, zip(*cluster2List)))
            cluster2ListMeanRound = (int(round(cluster2ListMean[0],0)), int(round(cluster2ListMean[1],0)))
            newCenterCluster2 = cluster2ListMeanRound

            # Check if we should break the algorithm
            if (centerCluster1 == newCenterCluster1) and (centerCluster2 == newCenterCluster2):
                # print "Breaking the algorithm after the following iterations:", iterationCounter
                break
            else:
                # Update the center
                centerCluster1 = newCenterCluster1
                centerCluster2 = newCenterCluster2
        return cluster1List, cluster2List

    def getROI(self, contours, imgSeedAndSprout, imgSprout, classStamp):
        # Run through all the contours

        #Debugging! Remove a lot of the white pixels, for better check if the cluster algoritm works...
        # DEBUGimgSeedAndSprout = cv2.imread("/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/imgSeedAndSproutDEBUG.png", cv2.CV_LOAD_IMAGE_COLOR)
        # DEBUGimgSeedAndSprout = cv2.cvtColor(DEBUGimgSeedAndSprout, cv2.COLOR_BGR2GRAY)
        imgDraw = imgSeedAndSprout.copy()
        imgDraw = cv2.cvtColor(imgDraw, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            p1 = (x, y)
            p2 = (x+width, y)
            p3 = (x+width, y+height)
            p4 = (x, y+height)

            # Debugging: Drawing the boundingBox for each contour.
            # self.drawBoundingBox(p1, p2, p3, p4, self.imgContours, (255, 0, 0), 1)
            # cv2.imshow("Cropped ROI "+str(indexCounter), imgBBcropped)
            # indexCounter = indexCounter + 1

            # Crop out each boundingbox
            imgBBcropped = imgSeedAndSprout[y:y+height, x:x+width]
            imgBBcroppedSprout = imgSprout[y:y+height, x:x+width]
            # cv2.imshow("Show the imgBBcropped", imgBBcropped)
            # cv2.imshow("Show the imgBBcropped sprout only", imgBBcroppedSprout)
            # cv2.imwrite("/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/imgSeedAndSprout.png", imgSeedAndSprout)

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

            # Now here find out which cluster/group this pixel coordinate belongs to.
            # Either it is the sprout cluster or the non-sprout cluster. It can happens that some pixels in the HSV segmentation, which
            # clearly not belongs to the sprout pixel somehow was continued through the HSV segmentation and morphology.
            # It is assumed that the real sprout pixels are clustered together and the non are more spread out.
            # Also it is assumed that the real sprout pixels shares more pixel coordinates compaired to the non-sprout pixels.
            # Some seed will have "harepix" on their body, which in this case is not handled and will result in false features.
            # However the main picture is that real sprout pixels belongs are neighbors and those pixel that are not sprout pixels are in some distance
            # more away from the real sprout pixels.

            if sprout:

                # Here we insert the check to see if the sprout are clustered together in one cluster or we have a main cluster with additionally noice blobs
                # In order to check how many clusters there is, the connect component is used, which is implemented in the findContours function
                # From here, the length of contours tells how many conected component there is in the image.
                imgBBcroppedSproutForFindContours = imgBBcroppedSprout.copy()
                blobs, hierarchy = cv2.findContours(imgBBcroppedSproutForFindContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print "So the number of blobs in this contour is:", len(blobs)
                # cv2.imshow("Show the imgBBcroppedSprout", imgBBcroppedSprout)
                # cv2.waitKey(0)

                if len(blobs) != 1:
                    # Run the K-means algorithm and clustering the list of sprout pixels into K clusters.
                    # cluster1List, cluster2List = self.runKmediansAlgorithm(sprout=sprout)
                    cluster1List, cluster2List = self.runKmeansAlgorithm(sprout=sprout)
                    # print "Done with the K-means algorithm and the cluster1List is:", cluster1List, "with a length of:", len(cluster1List)
                    # print "Done with the K-means algorithm and the cluster2List is:", cluster2List, "with a length of:", len(cluster2List)

                    # In order to chose the right list, we assume the following stuff:
                    # The list, that contains the true sprout pixels, must be the list, that has most pixel inside
                    # Perhaps also some with variance, but this is not important right now.
                    sprout = self.getLongestList(cluster1List, cluster2List)

                # Then convert it, in order to let it be used with the minAreaRect function
                sproutConvertedFormat = self.convertFormatForMinRectArea(sprout)

                # Then find the oriented bounding box of the sprouts
                obbSprout = cv2.minAreaRect(sproutConvertedFormat)

                # From here we do some feature extraction
                # Perhaps this must be wrapped into an featureextraction function
                # Anyway...

                # Then make sure that the "length" is the longest side of the boundingbox
                # and the "width" is the shortest side of the boundingbox.
                length, width = self.getLengthAndWidthFromSprout(obbSprout)
                p1, p2, p3, p4 = self.getBoxPoints(obbSprout)

                # Here we have to store the length, and width. Thease are the features from the sprout bounding box.


                # Debug: Convert the imgSeedAndSproutImage to an color image, in order to draw color on it
                # imgDraw = imgSeedAndSprout.copy()
                # imgDraw = cv2.cvtColor(imgDraw, cv2.COLOR_GRAY2BGR)
                self.drawBoundingBox(p1, p2, p3, p4, imgDraw, (0, 0, 255), 1)

                # cv2.drawContours(mask, contours, -1, contourColor, lineWidth)
                # cv2.imshow("imgContours"+str(classStamp), imgDraw)
                # cv2.waitKey(0)
            else:
                print "Hey this contour contained no sprout pixels from classstamp", classStamp

        # Show result at the end of all contours has been ran through...
        cv2.imshow("imgContours"+str(classStamp), imgDraw)
        cv2.imwrite("/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/imgRGB"+str(classStamp)+".png", self.imgRGB)
        cv2.imwrite("/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/imgSeedAndSprout"+str(classStamp)+".png", imgSeedAndSprout)
        cv2.imwrite("/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/imgContours"+str(classStamp)+".png", imgDraw)
        # cv2.waitKey(0)

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
        cv2.line(imgDraw, p1, p2, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p2, p3, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p3, p4, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p4, p1, boundingBoxColor, lineWidth)
