# -*- coding: utf-8 -*-



"""
Created on 30/11-2014

@author: christian
"""

##########################################
# Libraries
##########################################
import numpy as np                  # required for calculate i.e mean with np.mean
from pexpect import searcher_re
from scipy.maxentropy.maxentropy import conditionalmodel
import cv2                          # required for use OpenCV
import matplotlib.pyplot as plt     # required for plotting
import matplotlib
import pylab                        # required for arrange doing the wx list
import random                       # required to choose random initial weights and shuffle data
from PIL import Image
from planar import BoundingBox      # Required to use https://pythonhosted.org/planar/bbox.html

##########################################
# Classes
##########################################

class ProcessImage(object):

    #The constructor will run each time an object is assigned to this class.
    def __init__(self, img, classStamp):
        # Store which class the image is stamped
        self.classStamp = classStamp

        # Store the image argument into the public variable img
        self.img = img

        #########################################################
        # Initial trackbar vailes
        #########################################################
        # Set the threshold value to 128, in order to have gray pixels for sprouts and seeds and black pixels for background.
        self.thresholdValue = 128
        self.minHue = 27
        self.maxHue = 180
        self.minSaturation = 0
        self.maxSaturation = 255
        self.minValue = 147
        self.maxValue = 255
        self.minContourArea = 200
        self.maxContourArea = 2000

        # Test with RGB
        # self.minR = 0
        # self.minG = 0
        # self.minB = 0
        # self.maxR = 255
        # self.maxG = 255
        # self.maxB = 255

        # Name of windows
        nameOfHSVwindow = "The HSV segmentation with trackbar"
        nameOfThresholdWindow = "The thresholding with trackbar"
        nameOfContourWindow = "The contours with trackbar"

        #########################################################
        # Initial trackbar vailes
        #########################################################

        # Very bad coding... while(1) inside a constructor? Never mind...
        # However used to readjust the HSV, threshold, minMax area in countours before the program continues...
        # img = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/pauseScreen.jpg", cv2.CV_LOAD_IMAGE_COLOR)

        #########################################################
        # Udating the trackbar values
        #########################################################
        # self.lower_hsv = np.array([self.minHue, self.minSaturation, self.minValue], dtype=np.uint8)
        # self.upper_hsv = np.array([self.maxHue, self.maxSaturation, self.maxValue], dtype=np.uint8)
        # self.lower_rgb = np.array([self.minR, self.minG, self.minB], dtype=np.uint8)
        # self.upper_rgb = np.array([self.maxR, self.maxG, self.maxB], dtype=np.uint8)

        #########################################################
        # Add trackbars
        #########################################################
        self.addTrackbar("Threshold", nameOfThresholdWindow, self.thresholdValue, 255)
        self.addTrackbar("Min Hue", nameOfHSVwindow, self.minHue, 180)
        self.addTrackbar("Max Hue", nameOfHSVwindow, self.maxHue, 180)
        self.addTrackbar("Min Saturation", nameOfHSVwindow, self.minSaturation, 255)
        self.addTrackbar("Max Saturation", nameOfHSVwindow, self.maxSaturation, 255)
        self.addTrackbar("Min Value", nameOfHSVwindow, self.minValue, 255)
        self.addTrackbar("Max Value", nameOfHSVwindow, self.maxValue, 255)
        # self.addTrackbar("Min R", "The RGB segmentation with trackbar", self.minR, 255)
        # self.addTrackbar("Max R", "The RGB segmentation with trackbar", self.maxR, 255)
        # self.addTrackbar("Min G", "The RGB segmentation with trackbar", self.minG, 255)
        # self.addTrackbar("Max G", "The RGB segmentation with trackbar", self.maxG, 255)
        # self.addTrackbar("Min B", "The RGB segmentation with trackbar", self.minB, 255)
        # self.addTrackbar("Max B", "The RGB segmentation with trackbar", self.maxB, 255)
        self.addTrackbar("Min contour area", nameOfContourWindow, self.minContourArea, 10000)
        self.addTrackbar("Max contour area", nameOfContourWindow, self.maxContourArea, 100000)

        while True:
            k = cv2.waitKey(30) & 0xff
            # self.showImg("PauseScreen", img, 1)
            if k is 27:
                print("User closed the program...")
                break

            #########################################################
            # Listen to the trackbars
            #########################################################
            self.thresholdValue = self.trackbarListener("Threshold", nameOfThresholdWindow)
            self.minHue = self.trackbarListener("Min Hue", nameOfHSVwindow)
            self.maxHue = self.trackbarListener("Max Hue", nameOfHSVwindow)
            self.minSaturation = self.trackbarListener("Min Saturation", nameOfHSVwindow)
            self.maxSaturation = self.trackbarListener("Max Saturation", nameOfHSVwindow)
            self.minValue = self.trackbarListener("Min Value", nameOfHSVwindow)
            self.maxValue = self.trackbarListener("Max Value", nameOfHSVwindow)

            # Listen to the change of the parameters
            # self.minR = self.trackbarListener("Min R", "The RGB segmentation with trackbar")
            # self.maxR = self.trackbarListener("Max R", "The RGB segmentation with trackbar")
            # self.minG = self.trackbarListener("Min G", "The RGB segmentation with trackbar")
            # self.maxG = self.trackbarListener("Max G", "The RGB segmentation with trackbar")
            # self.minB = self.trackbarListener("Min B", "The RGB segmentation with trackbar")
            # self.maxB = self.trackbarListener("Max B", "The RGB segmentation with trackbar")
            self.minContourArea = self.trackbarListener("Min contour area", nameOfContourWindow)
            self.maxContourArea = self.trackbarListener("Max contour area", nameOfContourWindow)

            # Get the RGB image
            # self.imgRGB = self.getRGB(self.lower_rgb, self.upper_rgb)

            # Trying to use morphology to have a better thresholded image
            # NOTE: The image is offset in upper, right direction with the same size as the kernel size.
            # I.e. if using a 3,3, kernel, the morped image is shifted 3 times up and 3 times left.

            # Get the thresholded image and morph it
            kernelSize = 3
            self.imgThreshold = self.getThresholdImage()
            self.showImg("Check the thresholded image before morph", self.imgThreshold, 1)
            self.imgThreshold = self.getClosing(kernelSize, self.imgThreshold, 3, 3)
            self.showImg("Check the thresholded image after morph", self.imgThreshold, 1)

            # Get the HSV and morph it
            self.lower_hsv = np.array([self.minHue, self.minSaturation, self.minValue], dtype=np.uint8)
            self.upper_hsv = np.array([self.maxHue, self.maxSaturation, self.maxValue], dtype=np.uint8)

            self.imgHSV = self.getHSV(self.lower_hsv, self.upper_hsv)
            self.showImg("Showing the HSV before morph", self.imgHSV, 1)
            self.imgHSV = self.getClosing(kernelSize, self.imgHSV, 3, 3)          # Remove any noise pixels with erosion first and then dilate after (Opening)
            self.showImg("Showing the HSV after morph", self.imgHSV, 1)

            # Show original input image for compaire...
            self.showImg("Compair to input image", self.img, 1)

            # contours
            self.contoursFromThresholdImg = self.getContours(self.imgThreshold)  # Find the contours of the whole objects, to later do some matching...

            # Filter out the number of contours, like small noise-blobs, etc.
            self.contoursFromThresholdImgFiltered, listOfAreas = self.getContoursFilter(self.contoursFromThresholdImg, self.minContourArea, self.maxContourArea)

            #############################################
            # Try to crop out the first contour
            # This is done by finding the bounding box around a contour.
            #############################################

            # print "The contoursFromThreshold is:", self.contoursFromThresholdImg
            # rect = cv2.minAreaRect(self.contoursFromThresholdImg)
            # obbSeedAndSprout = self.getMinAreaRect(self.contoursFromThresholdImg)

            # p1, p2, p3, p4 = self.getBoxPoints(obbSeedAndSprout)
            # self.drawSproutBoundingBox(p1, p2, p3, p4)
            # self.showImg("Test the bounding box of the thresholded image", self.imgDrawings)

            #Print out the areas.
            # print listOfAreas
            # f = open('workfileClass' + str(classStamp) + '.txt', 'w')
            # f.write(str(listOfAreas))

            self.imgSeedAndSprout = self.addImg(self.imgThreshold, self.imgHSV) # Let the background be black,seeds be gray and sprouts be white
            self.showImg("Showing the imgSeedAndSprout after morph", self.imgSeedAndSprout, 1)

            # Draw the
            self.imgContourDrawing = self.imgThreshold.copy()
            # Set lineWidth to negative, to draw all the pixel within each contour.
            lineWidth = 2
            self.imgContourDrawing = self.drawContour(self.imgContourDrawing, self.contoursFromThresholdImgFiltered, lineWidth)
            self.showImg("Show contours of imgThreshold after morph", self.imgContourDrawing, 1)

            # In order to draw on the input image, without messing the original, a copy of the input image is made. This is called imgDrawings
            self.imgDrawings = self.img.copy()

            if self.contoursFromThresholdImgFiltered:
                print "So the contoursFromThresholdImgFiltered looks like this:", self.contoursFromThresholdImgFiltered
                self.features = self.getFeaturesFromContours(self.imgSeedAndSprout, self.contoursFromThresholdImgFiltered, self.classStamp) # The 100 is not testet to fit the smallest sprout

                # Draw the center of mass, on the copy of the input image.
                # Note: Somehow it seems there is a little offset in the input image, when the sprouts are longer.
                circleSize = 5
                self.drawCentroid(self.imgDrawings, self.features[1], circleSize, (0, 255, 255))

                # Write the hue_mean, hue_std number of sprout pixels, and the length, and width of the boundingBox around the each sprout
                # self.getTextForContours(self.imgDrawings, self.features[0])

            else:
                self.features = None

            # Show some figures....
            self.showImg(nameOfHSVwindow, self.imgHSV, 1)
            self.showImg(nameOfThresholdWindow, self.imgThreshold, 1)
            self.showImg(nameOfContourWindow, self.imgContourDrawing, 1)

            # self.showImg("The RGB segmentation with trackbar", self.imgRGB, 1)
            # cv2.imshow("The input image for compaire", self.img)

        print "Program restarted here..."

        # Clean up all the windows
        cv2.destroyWindow("PauseScreen")
        cv2.destroyAllWindows()

    def trackbarListener(self, nameOfTrackbar, nameOfWindow):
        value = cv2.getTrackbarPos(nameOfTrackbar, nameOfWindow)
        return value

    def addTrackbar(self, nameOfTrackbar, nameOfWindow, value, maxValue):
        cv2.namedWindow(nameOfWindow)
        cv2.createTrackbar(nameOfTrackbar, nameOfWindow, value, maxValue, self.nothing)

    def nothing(self, x):
        pass

    def getRatio(self, length, width):
        if (length or width) == 0:
            ratio = 0
        else:
            ratio = float(width/length)
        return ratio

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

    def getTextForContours(self, img, resultList):
        # print "We are inside the getTextForContours"
        numberOfDecimals = 2

        for element in resultList:
            self.drawText(img,                                      # The input image
                          element[0][0],                            # The x pixel COM position
                          element[0][1],                            # The y pixel COM position
                          round(element[1], numberOfDecimals),      # The hue_mean
                          round(element[2], numberOfDecimals),      # The hue_std
                          element[3],                               # The numberOfSproutPixels
                          round(element[4], numberOfDecimals),      # The length in pixels of the boundingBox of the sprout
                          round(element[5], numberOfDecimals),      # The width in pixels of the boundingBox of the sprout
                          round(element[6], numberOfDecimals)       # The ratio width/length
                          )

    def drawText(self, img, rows, cols, hue_mean, hue_std, numberOfSproutPixels, length, width, ratio):
        # print "The size of the image is:", img.shape
        offsetCols = 75   # 755
        offsetRows = 20   # 925

        # print "The dimension of the image is:", self.img.shape  # This was 660 x 920.

        textSize = 0.50
        textWidth = 1
        textColor = (0, 255, 0)

        # If the hue_mean is zero, then there was no sprout pixels and hence all the other features is zero.
        # In that case we dont want to put text on the image, if is is uninteressting.
        if hue_mean:
            cv2.putText(img, "mu:" + str(hue_mean), (rows-offsetCols,                       cols+offsetRows),   cv2.FONT_HERSHEY_SIMPLEX, textSize, textColor, textWidth)
            cv2.putText(img, "std:" + str(hue_std), (rows-offsetCols,                       cols+2*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, textSize, textColor, textWidth)
            cv2.putText(img, "Pixels:" + str(numberOfSproutPixels), (rows-offsetCols,       cols+3*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, textSize, textColor, textWidth)
            cv2.putText(img, "length:" + str(length), (rows-offsetCols,                     cols+4*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, textSize, textColor, textWidth)
            cv2.putText(img, "width:" + str(width), (rows-offsetCols,                       cols+5*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, textSize, textColor, textWidth)
            cv2.putText(img, "ratio:" + str(ratio), (rows-offsetCols,                       cols+6*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, textSize, textColor, textWidth)

    def drawContour(self, img, contours, lineWidth):
        # img_copy = self.imgThreshold.copy()
        # mask = np.zeros(img_copy.shape, dtype="uint8")
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(mask, self.contoursThreshold, -1, (0,255,0), 2)
        # self.showImg("Drawing the contours", mask, 0.5)


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

        # addedMask = self.addImg(self.lastMask, mask)
        # self.lastMask = addedMask

    def convertRGB2Hue(self, r, g, b):
        # http://www.rapidtables.com/convert/color/rgb-to-hsv.htm
        r_temp = float(r)/255
        g_temp = float(g)/255
        b_temp = float(b)/255
        c_max = max(r_temp, g_temp, b_temp)
        c_min = min(r_temp, g_temp, b_temp)
        delta = c_max-c_min

        if not delta:
            # print "r_temp:", r_temp
            # print "r:", r
            # print "g_temp:", g_temp
            # print "g:", g
            # print "b_temp:", b_temp
            # print "b:", b
            hue = 0
            return hue

        if c_max == r_temp:
            hue = 60 * (((g_temp-b_temp)/delta) % 6)
        elif c_max == g_temp:
            hue = 60 * (((b_temp-r_temp)/delta) + 2)
        elif c_max == b_temp:
            hue = 60 * (((r_temp-g_temp)/delta) + 4)
        else:
            print "Debug. Should not get into this else"
        return hue

    def analyseSprouts(self, sproutPixels):
        hue_temp = []
        numberOfSproutPixels = len(sproutPixels)

        for pixel in sproutPixels:
            r = self.img[pixel][2]
            g = self.img[pixel][1]
            b = self.img[pixel][0]

            x = pixel[1]   # The cols
            y = pixel[0]   # The rows
            # print "The pixel coordinate is: ", pixel, "and the RGB value at the given coordinate is: ", self.img[pixel], "which is in hue:", convertRGB2Hue(self.img[pixel])
            # print "This sprout pixel coordinate is: (", x, ",", y, ") and the RGB value at the given coordinate is r:", r, "g:", g, "b:", b, "which is in hue:", self.convertRGB2Hue(r,g,b)
            hue_temp.append(self.convertRGB2Hue(r,g,b))

        # Here calculate the mean hue
        hue_mean = np.mean(hue_temp)
        hue_std = np.std(hue_temp)
        # print "hue_mean is:", hue_mean
        # print "hue_std is:", hue_std

        return hue_mean, hue_std, numberOfSproutPixels

    def getSproutAndSeedPixels(self, img, contour):
        seeds = []
        sprouts = []

        for pixel in contour:
            # But when we have to get the intensity value out, the index of an pixel is called like this:
            # img[rows,cols], so we have to swop, since the (x,y) coordinate in OpenCV read image is (cols, rows)
            # Therefore we swop, in order to make it right.
            row = pixel[0][1]
            col = pixel[0][0]

            # NOTE: It does not make sense to say pixel[1][0], since we examinate only one pixel at a time, which is on element 0.
            # and on this element there is a row and a colum value.

            if img[row, col] == 255:
                #print "it is a white pixel, since the value is: ", img[rows, cols]
                sprouts.append((row, col))

            elif img[row, col] == 128:
                #print "it is a gray pixel, since the value is: ", img[rows, cols]
                seeds.append((row, col))
            else:
                print "it is a black pixel , since the value is: ", img[row, col]
                print "Should not get here, since the list of contours, contourThreshold only contains the seed and sprouts and not any background contour"

        return sprouts, seeds

    def drawSproutBoundingBox(self, p1, p2, p3, p4):
        # Draw the oriente bouningbox
        lineWidth = 1

        if self.classStamp is 1:
            boundingBoxColor = (0, 0, 255)
            cv2.line(self.imgDrawings, p1, p2, boundingBoxColor, lineWidth)
            cv2.line(self.imgDrawings, p2, p3, boundingBoxColor, lineWidth)
            cv2.line(self.imgDrawings, p3, p4, boundingBoxColor, lineWidth)
            cv2.line(self.imgDrawings, p4, p1, boundingBoxColor, lineWidth)

        elif self.classStamp is -1:
            boundingBoxColor = (255, 0, 0)
            cv2.line(self.imgDrawings, p1, p2, boundingBoxColor, lineWidth)
            cv2.line(self.imgDrawings, p2, p3, boundingBoxColor, lineWidth)
            cv2.line(self.imgDrawings, p3, p4, boundingBoxColor, lineWidth)
            cv2.line(self.imgDrawings, p4, p1, boundingBoxColor, lineWidth)
        else:
            pass # Dont want to have the green bounding box at the moement....
            # In case this looks ugly... I agree. But I was too lazy to make a function that draws those lines
            # I just wanted to be able to outcomment drawing the boundingbox for any of the classes.
            # boundingBoxColor = (0, 255, 0)
            # cv2.line(self.imgDrawings, p1, p2, boundingBoxColor, lineWidth)
            # cv2.line(self.imgDrawings, p2, p3, boundingBoxColor, lineWidth)
            # cv2.line(self.imgDrawings, p3, p4, boundingBoxColor, lineWidth)
            # cv2.line(self.imgDrawings, p4, p1, boundingBoxColor, lineWidth)

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

    def getFeaturesFromContours(self, img, contours, classStamp):
        # Define the list of elements
        elementList = []

        # Define the list of values for all the contours for each element.
        resultList = []
        seedCenterList = []
        hueMeanList = []
        hueStdList = []
        numberOfSproutPixelsList = []
        lengthList = []
        widthList = []
        ratioList = []
        classStampList = []

        #Run through each contour in the contour list (contours) and check each pixel in that contour.
        seedCounter = 0
        sproutCounter = 0
        # print "--------------------- Running trough all the contours ---------------------"
        # print ""

        for contour in contours:
            # Now we are at the first contour that contains a lot of pixel coordinates, i.g. (x1,y1), (x2,y2),...,(xn,yn)
            seedCounter = seedCounter + 1

            # print "This contour contains", len(contour), "pixels coordinates" # So this is how many pixel sets (x,y) there is in each contour.

            #Then we check each pixel coordinate in this contour
            # and compair it with the same pixel intensity in the input img image.

            # For each contour, the sprout and seed pixels is founded and stored in the sprouts and seeds lists.
            # self.showImg("We are gettign the seed and sprout pixels by looking at this image:", img, 0.5)
            # cv2.waitKey(0)

            sprout, seed = self.getSproutAndSeedPixels(img, contour)
            # print "The seed is:", seed
            # print "The sprout is:", sprout

            # If this contour contains any sprout pixels
            if sprout:

                sproutCounter = sproutCounter + 1
                # Then convert it, in order to let it be used with the minAreaRect function
                sproutConvertedFormat = self.convertFormatForMinRectArea(sprout)

                # Then find the oriented bounding box
                obbSprout = self.getMinAreaRect(sproutConvertedFormat)
                length, width = self.getLengthAndWidthFromSprout(obbSprout)
                p1, p2, p3, p4 = self.getBoxPoints(obbSprout)
                self.drawSproutBoundingBox(p1, p2, p3, p4)

                # Get the width/length ratio
                ratio = self.getRatio(length, width)

                # Get the hue_mean, hue_std and number of sprout pixels
                hue_mean, hue_std, numberOfSproutPixels = self.analyseSprouts(sprout)

                # Debugging
                # self.drawContour("Testing the contour", img, sproutConvertedFormat, 0.9)

                # Try to draw each sprout in one color and the seed in an other color to verify that the sprout and seed list
                # really contains the pixels that belongs to a sprout or and seed.

            # Else the sprout pixel list was empty, and hence the length, width and ratio is set to zero
            else:
                length = 0
                width = 0
                ratio = 0
                hue_mean = 0
                hue_std = 0
                numberOfSproutPixels = 0

            # Then convert it, in order to let it be used with the minAreaRect function
            # seedConvertedFormat = self.convertFormatForMinRectArea(seed)
            # Then find the oriented bounding box
            # obbSeed = self.getMinAreaRect(seedConvertedFormat)
            # print "Now the obbSeed is:", obbSeed

            # Append the seed pixels into a temp_array in order to find the COM
            temp_array = []
            temp_array.append(seed)

            # Check if temp_array is not-empty
            if len(seed):
                center_of_mass = self.getCentroidOfSingleContour(temp_array)
            else:
                center_of_mass = 0, 0

            # Compairing the minAreaRect centercoordinate output with the moments center coordinate
            # print "The center_of_mass is:", center_of_mass
            # print "Comapired to the other center of mass:", obbSeed[0]
            # x = int(obbSeed[0][0])
            # y = int(obbSeed[0][1])
            # cv2.circle(self.imgDrawings, (x,y), 4, (0, 0, 255), -1)
            # self.showImg("Testing drawing", self.imgDrawings, 0.5)
            elementList.append(center_of_mass)

            # In order to get all the pixels here, all the coordinates inside a given countour must be
            # found.

            # hue_mean, hue_std, numberOfSproutPixels = self.analyseSprouts(sprout)
            elementList.append(hue_mean)
            elementList.append(hue_std)
            elementList.append(numberOfSproutPixels)
            elementList.append(length)
            elementList.append(width)
            elementList.append(ratio)
            elementList.append(classStamp)

            # Finally append the element list into the resultList
            resultList.append(elementList)

            #Append to all the feature lists in order to plot these feature later on a feature space diagram.
            seedCenterList.append(center_of_mass)
            hueMeanList.append(hue_mean)
            hueStdList.append(hue_std)
            numberOfSproutPixelsList.append(numberOfSproutPixels)
            lengthList.append(length)
            widthList.append(width)
            ratioList.append(ratio)
            classStampList.append(classStamp)

            # Clear the element list for each contour
            elementList = []

            # print "-----------------------Done with that given contour --------------------------------------------"
            # print ""

        # print "seedCounter is", seedCounter
        # print "sproutCounter is", sproutCounter
        # print "--------------------- DONE with the contours --------------------"
        # print ""
        return resultList, seedCenterList, hueMeanList, hueStdList, numberOfSproutPixelsList, lengthList, widthList, ratioList, classStampList

    def showImg(self, nameOfWindow, image, scale):

        imgCopy = image.copy()
        image_show = self.scaleImg(imgCopy, scale)
        cv2.imshow(nameOfWindow, image_show)
        cv2.imwrite("/home/christian/workspace_python/MasterThesis/SeedDetection/writefiles/" + str(nameOfWindow) + ".png", image_show)

    def getROI(self, image, startY, endY, startX, endX):
        roi = image.copy()
        return roi[startY:endY, startX:endX]

    def scaleImg(self, image, scale):
        img_scale = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return img_scale

    def subtractImg(self, image1, image2):
        subtractImage = cv2.subtract(image1, image2)
        return subtractImage

    def addImg(self, image1, image2):
        addedImage = cv2.add(image1, image2)
        return addedImage

    # def getGrayImage(self):
    #
    #     img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    #     return img_gray

    def getThresholdImage(self):
        #Do the grayscale converting
        imgGray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Do the thresholding of the image.
        ret, img_threshold = cv2.threshold(imgGray, self.thresholdValue, 128, cv2.THRESH_BINARY)

        return img_threshold

    def getEdge(self):
        img_edges = cv2.Canny(self.imgGray, self.minCannyValue, self.maxCannyValue)
        return img_edges

    def getHSV(self, lower_hsv, upper_hsv):
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img_seg = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        return img_seg

    def getRGB(self, lower_rgb, upper_rgb):
        img_rgb = self.img
        img_seg = cv2.inRange(img_rgb, lower_rgb, upper_rgb)
        return img_seg

    def getContours(string, binary_img):
        #Copy the image, to avoid manipulating with original
        contour_img = binary_img.copy()

        #Find the contours of the thresholded image
        #Note: See OpenCV doc if needed to change the arguments in findContours.
        # Note: The CHAIN_APPROX_SIMPLE only takes the outer coordinates. I.e. a square has only four coordinates instead
        # of following the edge all around. The CHAIN_APPROX_NONE takes all the pixels around the edge
        contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # print "The contour is:", contours
        #
        # print "Now trying to use minAreaRect within the getContours function"
        # rect = cv2.minAreaRect(contours)
        # print "It worked!!!"

        #Return the contours. We dont want to use the hierarchy yet
        # However the hierarchy is usefull the detect contours inside a contour or dont detect it.
        # That is what hierarchy keeps track of. (Children and parents)
        return contours

    def getCentroidOfSingleContour(self, contour):

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
        # center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) # I swoped it here, since the coordinate was swopped..
        center = (int(m['m01'] / m['m00']), int(m['m10'] / m['m00']))

        return center

    def getCentroid(self, contours, areaThreshold):
        # print "Inside getCentroid: ",  contours

        # List of centers
        centers = []

        #Run through all the contours
        for contour in contours:

            #Get the area of each contour
            contour_area = cv2.contourArea(contour, False)

            if (contour_area < areaThreshold) or (contour_area > areaThreshold + 5000):
                # print "The contour area is", contour_area, "and hence skipped"
                continue

            #Calculate the moments for each contour in contours
            m = cv2.moments(contour)

            #If somehow one of the moments is zero, then we brake and reenter the loop (continue)
            #to avoid dividing with zero
            if (int(m['m01']) == 0 or int(m['m00'] == 0)):
                continue

            #Calculate the centroid x,y, coordinate out from standard formula.
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

            #Append each calculated center into the centers list.
            centers.append(center)
        return centers

    def drawCentroid(self, image, centers, size, RGB_list):
        # Color the central coordinates for red bricks with a filled circle
        for center in centers:
            cv2.circle(image, center, size, RGB_list, -1)

    def getErode(self, img_binary, iterations_erode):
        kernel = np.ones((3, 3), np.uint8)
        img_erode = cv2.erode(img_binary, kernel, iterations=iterations_erode)
        return img_erode

    def getDilate(self, img_binary, iterations_dilate):
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_binary, kernel, iterations=iterations_dilate)
        return img_dilate

    def getOpening(self, kernelSize, img_binary, iterations_erode, iterations_dilate):
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
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
        crop = img_morph[0:height-kernelSize+1, 0:width-kernelSize+1]
        img_morph = cv2.copyMakeBorder(crop, kernelSize-1, 0, kernelSize-1, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_morph

    def getClosing(self, kernelSize, img_binary, iterations_erode, iterations_dilate):
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
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
        crop = img_morph[0:height-kernelSize+1, 0:width-kernelSize+1]
        img_morph = cv2.copyMakeBorder(crop, kernelSize-1, 0, kernelSize-1, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_morph

    def nothing(self, x):
        pass

    def addTrackbar(self, name, nameOfWindow, startValue, maxValue):
        cv2.namedWindow(nameOfWindow)
        cv2.createTrackbar(name, nameOfWindow, startValue, maxValue, self.nothing)

    def trackbarListener(self, name, nameOfWindow):
        value = cv2.getTrackbarPos(name, nameOfWindow)
        return value

    def getMinAreaRect(self, list_np):
        rect = cv2.minAreaRect(list_np)
        return rect

    def getBoxPoints(self, rect):
        # http://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        p1 = (box[0][0], box[0][1])
        p2 = (box[1][0], box[1][1])
        p3 = (box[2][0], box[2][1])
        p4 = (box[3][0], box[3][1])
        return p1, p2, p3, p4

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
        print "Now the list_np looks like this", list_np
        return list_np

    def getFeatureLabel(self, featureIndex):
        if featureIndex == 2:
            featureLabel = 'hue_mean'
        elif featureIndex == 3:
            featureLabel = 'hue_std'
        elif featureIndex == 4:
            featureLabel = 'number of sprout pixels'
        elif featureIndex == 5:
            featureLabel = 'length'
        elif featureIndex == 6:
            featureLabel = 'width'
        elif featureIndex == 7:
            featureLabel = 'width/length ratio'
        else:
            featureLabel = 'No valid features label'
        return featureLabel

class PlotFigures():
    def __init__(self, titleName, fileName):
        self.fileName = fileName

        self.size = 18
        font = {'size': self.size}
        matplotlib.rc('xtick', labelsize=self.size)
        matplotlib.rc('ytick', labelsize=self.size)
        matplotlib.rc('font', **font)

        # self.fig = plt.figure(num=titleName, figsize=(10.94, 8.21), dpi=200, facecolor='w', edgecolor='k')
        self.fig = plt.figure(num=titleName, figsize=(10, 8.21), dpi=300, facecolor='w', edgecolor='k')
        plt.title(titleName)
        self.ax = plt.subplot(111)

    def plotData(self, x, y, string_icon, string_label):
        plt.plot(x, y, string_icon, label=string_label, markersize=self.size/2)

        # Shrink current axis by 20%
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # Put a legend to the right of the current axis
        # self.ax.legend(loc='center left', bbox_to_anchor=(0.8, 1))
        self.ax.legend(loc='lower right')

        #Set grid on, limit the y axis (not the x yet) and put names on axis
        plt.grid(True)

    def setXlabel(self, string_x):
        plt.xlabel(string_x, fontsize=self.size)

    def setYlabel(self, string_y):
        plt.ylabel(string_y, fontsize=self.size)

    def limit_y(self, min_y, max_y):
        plt.ylim(min_y, max_y)

    def limit_x(self, min_x, max_x):
        plt.xlim(min_x, max_x)

    def plotMean(self, x, y, string_icon):
        plt.plot(np.mean(x), np.mean(y), string_icon, markersize=20)

    def updateFigure(self):
        plt.show(block=False)   # It is very big with 300 dpi
        self.saveFigure()

    def saveFigure(self):
        # plt.annotate('Removed datapoint', xy=(0.33, 0.43), xytext=(0.6, 0.5), arrowprops=dict(facecolor='black', shrink=0.005))
        plt.savefig("/home/christian/workspace_python/MasterThesis/SeedDetection/writefiles/" + str(self.fileName) + ".png")

class Perceptron():
    def __init__(self):
        #Initial random weights and bias from 0.0 to 1.0
        #self.w = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
        #self.b = random.uniform(0.0, 1.0)

        self.w = [0.00001, 0.00001]
        self.b = 0.00001
        self.error = 0
        self.run_flag = True
        self.true_counter = 0

        # For plotting
        self.w0_plot = []
        self.w1_plot = []
        self.b_plot = []
        self.error_plot = []

        self.wx = 0
        self.wy = 0

        # For the classificaton
        self.class1 = []
        self.classNeg1 = []
        self.class1_featureX = []
        self.class1_featureY = []
        self.classNeg1_featureX = []
        self.classNeg1_featureY = []

    def startLearn(self, learning_rate, total_training_data):
        # Shuffle the total_training_data before the system starts learning.
        np.random.shuffle(total_training_data)

        print("Now the perceptron starts...")

        #Start the algorithm. RunFlag is already True
        self.true_counter = 0
        while self.run_flag == True:
            self.true_counter += 1
            #print('-' * 60)
            error_count = 0
            for data in total_training_data:
                # print "Now the dataX is:", data[featureX]
                # print "Now the dataY is:", data[featureY]
                # print("The weights is:", self.w)
                #Calculate the dotproduct between input and weights
                dot_product = data[0]*self.w[0] + data[1]*self.w[1]
                # print "And the dot_product is:", dot_product

                #If the dotprodcuct + the bias is >= 0, then result is class 1
                # else it is class -1.
                if dot_product + self.b >= 0:
                    result = 1
                else:
                    result = -1

                #Calculate error, where data[2] is the contourClass/desired output
                self.error = data[2] - result

                #Continue the while, continue the algorithm if only the error is not zero
                if self.error != 0:
                    error_count += 1
                    #Update the final waits and bias
                    self.w[0] += data[0]*learning_rate*self.error
                    self.w[1] += data[1]*learning_rate*self.error
                    self.b += learning_rate * self.error

                #Store the weights and bias
                self.w0_plot.append(self.w[0])
                self.w1_plot.append(self.w[1])
                self.b_plot.append(self.b)
                self.error_plot.append(self.error)

            if error_count == 0:
                # print("Now there is no errors in the whole trainingData")
                self.run_flag = False
                print("The number of iterations before the Perceptron stops is:", self.true_counter)

            tries = 1000
            if self.true_counter > tries:
                print "The Perceptron has run more than", tries, "times... So we abort now"
                self.run_flag = False

    def getClassifier(self, xmin, xmax, step):
        self.wx = pylab.arange(xmin, xmax, step)
        self.wy = (self.w[0]*self.wx)/(-self.w[1]) + (self.b)/(-self.w[1])

    def doClassification(self, testingData, centerSeedList, finalImage):
        #With the ready Perceptron classifier, we can now classify the testing data
        # and mark that on the original testing image.

        # Convert it to a numpy array
        np_testingdata = np.array(testingData, dtype=np.float)

        #Doing the classification. So if the y is negative, it belongs to class -1
        # and if the y is positive it belongs to class 1.
        # Before the testingData is intered the classifier, the data[2] = 0 --> unclassified.
        # After this for loop the data[2] is either -1 or +1

        for index, centerSeedList in zip(np_testingdata, centerSeedList):

            y = index[0]*self.w[0] + index[1]*self.w[1] + self.b
            if y >= 0:
                index[2] = 1
                cv2.circle(finalImage, centerSeedList, 5, (0, 0, 255), -1)
            else:
                index[2] = -1
                cv2.circle(finalImage, centerSeedList, 5, (255, 0, 0), -1)

        return np_testingdata

    def normalizeData(self, list):
        maxValueOfList = max(list)
        return np.array(list)/float(maxValueOfList)

    def getTotalList(self, list1, list2, list3):
        outList = []
        for elementInList1, elementInList2, elementInList3 in zip(list1, list2, list3):
            outList.append((elementInList1, elementInList2, elementInList3))
        return outList

    def getIndividualList(self, totalList):
        class1ListX = []
        class1ListY  = []
        classNeg1ListX = []
        classNeg1ListY = []
        for element in totalList:
            if int(element[2]) is 1:
                class1ListX.append(element[0])
                class1ListY.append(element[1])
            elif int(element[2]) is -1:
                classNeg1ListX.append(element[0])
                classNeg1ListY.append(element[1])
            else:
                print "Should not get down to this else"
        return class1ListX, class1ListY, classNeg1ListX, classNeg1ListY

class ProcessVideo(ProcessImage):
    #The constructor will run each time an object is assigned to this class.
    def __init__(self, string):
        self.cap = cv2.VideoCapture(string)
        self.img = self.getFrame()

    def getFrame(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            return self.frame
        else:
            print "Cant open video"

def main():

    ###############################################################################################
    # Generel note:
    #
    # The image, seed1, is an ideal image taking with flash from a high resolution camera. Using images from movie, will not be as nice
    # However I will start implementing the classification system based on a the nice image, and later we can focus on getting better images.
    # Using the PIL library  be able to see the size of the image.
    # im = Image.open("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed1.jpg")
    # print "size is: ", im.size
    # However the OpenCV does this also with: print "The shape of the image is: ", img.shape # rols x cols
    #
    # The next step is to interface a camera into the code.
    # And then each frame from the camera will be treated as a input_image
    #
    # seed31Mix.jpg             # Rækkefølgen er:  nede fra og op som billedet vender nu: (lidt) for gul- for lange- gode- gode- uspirede.
    # seed32ForLang.jpg
    # seed33GodeSpirer.jpg
    # seed34Uspiretfroe.jpg
    # seed35GulSpire.jpg
    # seed36SkadetSpire.jpg
    #
    # Overview of featuers
    # features[0] # The whole resultList is stored. I.e. [[(x,y), hue_mean, hue_std... classStamped],... [..]]
    # features[1] # Center seed coordinates
    # features[2] # Hue_mean
    # features[3] # Hue_std
    # features[4] # Number of sprout pixels
    # features[5] # Length
    # features[6] # Width
    # features[7] # Ratio of width/length
    # features[8] # classStamp   --> Not really a feature.
    ################################################################################################

    image_show_ratio = 1
    # Training data class 1. Define the first testing data set as class 1

    ###################################################
    # Image which was taken the 4/3-2015 at ImproSeed #
    ###################################################
    # imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section5/InputClass1.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClassNeg1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section5/InputClassNeg1.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTestingData = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section5/InputClass0.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    imgTrainingClassNeg1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_lang_og_krum.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    imgTestingData = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_Mix.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/td1.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClassNeg1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/tdNeg1.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTestingData = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/td0.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # imgTrainingClass1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/tooLong.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClass1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/seed32ForLangManipulated.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # imgTrainingClass1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/seed32ForLang.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClass1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/seedMix.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/Parametre4/par4test.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # Training data class -1. Define the secound testing data set as class -1
    # imgTrainingClassNeg1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/seed33GodeSpirer.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # imgTrainingClassNeg1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/seed33GodeSpirerManipulated.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClassNeg1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/tooShort.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTrainingClassNeg1 = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/day8_ImageCropped.jpg", cv2.CV_LOAD_IMAGE_COLOR)


    # Testing data class 0.
    # imgTestingData = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/seedMix.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # imgTestingData = cv2.imread("/home/christian/workspace_python/MasterThesis/SeedDetection/readfiles/seed31Mix.jpg", cv2.CV_LOAD_IMAGE_COLOR)


    # Do the image processing on the training data class 1
    print "--------------- Now doing the td1 ---------------"
    td1 = ProcessImage(imgTrainingClass1, 1)

    # Do the image processing on the training data class -1.
    print "--------------- Now doing the tdNeg1 ---------------"
    tdNeg1 = ProcessImage(imgTrainingClassNeg1, -1)   #Uncomment this one when all the debugging printout is removed.

    print "--------------- Now doing the testing data ---------------"
    # Do the image processing on the testing data class 0
    testData = ProcessImage(imgTestingData, 0)  #Uncomment this one when all the debugging printout is removed.

    # Make sure that there is available features in the training and testing data
    if td1.features and tdNeg1.features and testData:
        # features[0] # The whole resultList is stored. I.e. [[(x,y), hue_mean, hue_std... classStamped],... [..]]
        # features[1] # Center seed coordinates
        # features[2] # Hue_mean
        # features[3] # Hue_std
        # features[4] # Number of sprout pixels
        # features[5] # Length
        # features[6] # Width
        # features[7] # Ratio of width/length
        # features[8] # classStamp   --> Not really a feature.

        featureIndexX = 5
        featureIndexY = 4
        featureLabelX = td1.getFeatureLabel(featureIndexX)
        featureLabelY = td1.getFeatureLabel(featureIndexY)

        # Draw featurespace - not normalized
        drawData1 = PlotFigures("Feature space for training data class 1 and class -1", "FeatureSpaceClass1andClassNeg1")
        drawData1.plotData(td1.features[featureIndexX], td1.features[featureIndexY], "rs", "Class 1")
        drawData1.plotData(tdNeg1.features[featureIndexX], tdNeg1.features[featureIndexY], "bs", "Class -1")
        # drawData1.plotData(testData.features[featureIndexX], testData.features[featureIndexY], "gs", "Class 0") # Uncomment this just to see the mix data in the feature space
        drawData1.setXlabel(featureLabelX)
        drawData1.setYlabel(featureLabelY)
        drawData1.updateFigure()


        # Initialize the Perceptron, in order to get acces to the normalizeData function
        p = Perceptron()

        # Add the training data together, also with the testing data to insure propper normalization
        addedFeatureX = td1.features[featureIndexX] + tdNeg1.features[featureIndexX] + testData.features[featureIndexX]
        addedFeatureY = td1.features[featureIndexY] + tdNeg1.features[featureIndexY] + testData.features[featureIndexY]
        classStampList = td1.features[8] + tdNeg1.features[8] + testData.features[8]
        centerSeedList = testData.features[1]

        # Normalize the data here, now with the testing data as well.
        normalizedAddedFeatureX = p.normalizeData(addedFeatureX)
        normalizedAddedFeatureY = p.normalizeData(addedFeatureY)

        # For only getting the training data set, we have to get the testdata out of the normalizedAddedFeatureX and
        # normalizedAddedFeatureY. Since know that we stack the testing data at the end, then we just remove the last n elements
        # where n is the length of the testingdata.

        # Here we read the testing data only
        classZeroListX = normalizedAddedFeatureX[-len(testData.features[featureIndexX]):]
        classZeroListY = normalizedAddedFeatureY[-len(testData.features[featureIndexY]):]

        # Here we read the training data only
        normalizedAddedFeatureX = normalizedAddedFeatureX[0:-len(testData.features[featureIndexX])]
        normalizedAddedFeatureY = normalizedAddedFeatureY[0:-len(testData.features[featureIndexY])]

        # Create the list, that stores only the selected features, which was defined in line 905, 906
        total_training_data = p.getTotalList(normalizedAddedFeatureX, normalizedAddedFeatureY, classStampList)

        # Get the class1, classNeg1  lists in order to plot then with individual color.
        class1ListX, class1ListY, classNeg1ListX, classNeg1ListY = p.getIndividualList(total_training_data)

        #Run the Perceptron algorithm to learn the classifier something...
        learning_rate = 0.10
        p.startLearn(learning_rate, total_training_data)
        p.getClassifier(0, 1, 0.01)

        # Draw the data with the classifier line and with normalized data
        drawData2 = PlotFigures("Normalized feature space for class 1 and class -1 data \n with Perceptron classifier ", "NormFeatureSpaceClass1andClassNeg1WithPerceptron")
        drawData2.plotData(p.wx, p.wy, "b-", "The perceptron")
        drawData2.plotData(class1ListX, class1ListY, "rs", "Class 1")
        drawData2.plotData(classNeg1ListX, classNeg1ListY, "bs", "Class -1")
        drawData2.limit_y(0,1)
        drawData2.limit_x(0,1)
        # drawData2.plotData(classZeroListX, classZeroListY, "gs", "Class 0")
        drawData2.setXlabel(featureLabelX)
        drawData2.setYlabel(featureLabelY)
        drawData2.updateFigure()


        # Draw the data with the classifier line and the normalized testing data
        drawData3 = PlotFigures("Normalized feature space for testing data \n with Perceptron classifier ", "NormFeatureSpaceClass0WithPerceptron")
        drawData3.plotData(p.wx, p.wy, "b-", "The perceptron")
        drawData3.plotData(classZeroListX, classZeroListY, "gs", "Class 0")

        # print "The classZeroListX is:", classZeroListX
        # print "The classZeroListY is:", classZeroListY

        # Note: The legend is a little fucked... If there is only two elements, the Perceptron line and the testin data
        # that has not been classified, then the text of Perceptron goes out into the right margin of the image.
        # IF we add anohter data set, then the text aligns... HMm.
        drawData3.limit_y(0,1)
        drawData3.limit_x(0,1)
        drawData3.setXlabel(featureLabelX)
        drawData3.setYlabel(featureLabelY)
        drawData3.updateFigure()

        # print "The classZeroListX is:", classZeroListX
        # print "The classZeroListY is:", classZeroListY

        # In order to draw the blue and red color on the normalize data, we have to get the list of all
        # the elements which has class 1 and all the elements class -1. This is done after the training data is added
        # in order to insure propper normalization. If I do normalization before the training data is added,
        # then the normalization is not correct. It gives a little more work, but is the correct way to go.

        # Create the list, that stores only the selected features, which was defined in line 905, 906
        zeroList = np.zeros(len(testData.features[featureIndexX]), dtype=np.int)
        total_testing_data = p.getTotalList(classZeroListX, classZeroListY, zeroList)
        classifiedTestingData = p.doClassification(total_testing_data, centerSeedList, testData.imgDrawings)

        # Get the testing data which has been classfied as class1 or classNeg1 lists
        Finalclass1ListX, Finalclass1ListY, FinalclassNeg1ListX, FinalclassNeg1ListY = p.getIndividualList(classifiedTestingData)

        # Draw the data with the classifier line and where the testing data has been classified.
        drawData4 = PlotFigures("Normalized classified testing data", "NormClassifiedData")
        drawData4.plotData(p.wx, p.wy, "b-", "The perceptron")
        drawData4.plotData(Finalclass1ListX, Finalclass1ListY, "rs", "Class 1")
        drawData4.plotData(FinalclassNeg1ListX, FinalclassNeg1ListY, "bs", "Class -1")
        drawData4.limit_y(0,1)
        drawData4.limit_x(0,1)
        drawData4.setXlabel(featureLabelX)
        drawData4.setYlabel(featureLabelY)
        drawData4.updateFigure()

        ###############################################################################
        # Show some results
        ###############################################################################



        # Training data class 1
        td1.showImg("InputClass1", td1.img, image_show_ratio)

        # Training data class -1
        tdNeg1.showImg("InputClassNeg1", tdNeg1.img, image_show_ratio)

        # Testing data class 0
        testData.showImg("InputClass0", testData.img, image_show_ratio)

        # Show the grayscale image
        #td1.showImg("Grayscale image class 1 and fitted to display on  screen", td1.imgGray, image_show_ratio)
        # tdNeg1.showImg("Grayscale image class -1 and fitted to display on  screen", tdNeg1.imgGray, image_show_ratio)
        # testData.showImg("Grayscale image class 0 and fitted to display on  screen", testData.imgGray, image_show_ratio)

        # Show the thresholded image. This contains the seeds and the sprouts with a gray level. Background is black
        # td1.showImg("ThresholdedClass1", td1.imgThreshold, image_show_ratio)
        # tdNeg1.showImg("ThresholdedClassNeg1", tdNeg1.imgThreshold, image_show_ratio)
        # testData.showImg("ThresholdedClass0", testData.imgThreshold, image_show_ratio)

        # Show the contours which is the result of the findContours function from OpenCV
        # td1.showImg("ContoursTd1", td1.imgContourDrawing, image_show_ratio)
        # tdNeg1.showImg("ContoursTdNeg1", tdNeg1.imgContourDrawing, image_show_ratio)
        # testData.showImg("ContoursTestData", testData.imgContourDrawing, image_show_ratio)

        # Show the segmentated sprouts by using HSV
        # td1.showImg("HSVclass1", td1.imgHSV, image_show_ratio)
        # tdNeg1.showImg("HSVclassNeg1", tdNeg1.imgHSV, image_show_ratio)
        # testData.showImg("HSVclass0", testData.imgHSV, image_show_ratio)

        # Show the morph on the HSV to get nicer sprouts
        # td1.showImg("HSVclass1Morph", td1.imgMorph, image_show_ratio)
        # tdNeg1.showImg("HSVclassNeg1Morph", tdNeg1.imgMorph, image_show_ratio)
        # testData.showImg("HSVclass0Morph", testData.imgMorph, image_show_ratio)

        # Show the addition of the two images, thresholde and HSV
        # td1.showImg("AddedClass1", td1.imgSeedAndSprout, image_show_ratio)
        # tdNeg1.showImg("AddedClassNeg1", tdNeg1.imgSeedAndSprout, image_show_ratio)
        # testData.showImg("AddedClass0", testData.imgSeedAndSprout, image_show_ratio)

        # Show the input image with indicated center of mass coordinates of each seed.
        td1.showImg("ResultClass1", td1.imgDrawings, image_show_ratio)
        tdNeg1.showImg("ResultClassNeg1", tdNeg1.imgDrawings, image_show_ratio)
        testData.showImg("ResultClass0", testData.imgDrawings, image_show_ratio)

    else:
        if td1.features:
            print "No features available for tdNeg1"
        elif tdNeg1.features:
            print "No features available for td1"
        else:
            print "No features available for td1 and tdNeg1"

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # while(1):
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         print("User closed the program...")
    #         break
    # # Wait until the user hit any key
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # # This is the HSV segmentation
    # image.addTrackbar("Hue min", "HSV segmentation", image.minHue, 255)
    # image.addTrackbar("Hue max", "HSV segmentation", image.maxHue, 255)
    # image.addTrackbar("Saturation min", "HSV segmentation", image.minSaturation, 255)
    # image.addTrackbar("Saturation max", "HSV segmentation", image.maxSaturation, 255)
    # image.addTrackbar("Value min", "HSV segmentation", image.minValue, 255)
    # image.addTrackbar("Value max", "HSV segmentation", image.maxValue, 255)

    # Wait here, while user hits ESC.
    # while 1:
    #
    #     image_ratio = 0.2
    #     morph_iterations = 2
    #
    #     # Show the image
    #     image.showImg("Input image and fitted to display on screen", image.img, image_ratio)
    #
    #     # # This is the normal thresholding --- Test show that this was not useful in segmentating the eeeds vs. trout
    #     threshold_img = image.getThresholdImage(128)
    #     image.showImg("Threshold image", threshold_img, image_ratio)
    #     image.thresholdValue = image.trackbarListener("Threshold", "Threshold image")
    #
    #     # # This is the Canny edge detection  --- Test show that this was not very useful
    #     # image.showImg("Edge detection image", image.getEdge(), image_ratio)
    #     # image.minCannyValue = image.trackbarListener("Min value", "Edge detection image")
    #     # image.maxCannyValue = image.trackbarListener("Max value", "Edge detection image")
    #
    #     # # This is the HSV segmentation --- Test show that this was ?????????
    #     hsv_img = image.getHSV(image.lower_hsv, image.upper_hsv)
    #     # image.showImg("HSV segmentation", hsv_img, image_ratio)
    #     # image.lower_hsv[0] = image.trackbarListener("Hue min", "HSV segmentation")
    #     # image.upper_hsv[0] = image.trackbarListener("Hue max", "HSV segmentation")
    #     # image.lower_hsv[1] = image.trackbarListener("Saturation min", "HSV segmentation")
    #     # image.upper_hsv[1] = image.trackbarListener("Saturation max", "HSV segmentation")
    #     # image.lower_hsv[2] = image.trackbarListener("Value min", "HSV segmentation")
    #     # image.upper_hsv[2] = image.trackbarListener("Value max", "HSV segmentation")
    #
    #     image_dilate = image.getDilate(hsv_img, morph_iterations)
    #     # image.showImg("Dilate", image_dilate, image_ratio)
    #     image_erode = image.getErode(image_dilate, morph_iterations)
    #     # image.showImg("Closing", image_erode, image_ratio)
    #
    #     # Here add the grayscale and hsv image together. Note: use cv2.add otherwise pixelvale like 250 + 10 = 260 % 255 = 4, and not 255.
    #     seedAndTrout_img = cv2.add(threshold_img, hsv_img)
    #     image.showImg("Seed image", threshold_img, image_ratio)
    #     image.showImg("Trout image", hsv_img, image_ratio)
    #     image.showImg("Seed and trout image", seedAndTrout_img, image_ratio)
    #
    #     contours = image.getContours(threshold_img)
    #     centers = image.getCentroid(contours)
    #     image.drawCentroid(img, centers, (0, 255, 0))
    #
    #
    #
    #
    #
    #
    #
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         print("User closed the program...")
    #         break

    # Close down all open windows...

    # Wait for a user input to close down the script

    # cv2.destroyAllWindows()

        # imgTrainingClassNeg1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed33GodeSpirer.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # Now with testing data, which comes from a camera.
    # video = ProcessVideo(1)
    # video.showImg("The video input image", video.img, 1)
    # imgTrainingClassNeg1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed33GodeSpirer.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # cap = cv2.VideoCapture(2)
    # ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    #
    # print "Playing the input video - hit ESC to stop the video"
    # print "Is the cap open?", cap.isOpened()
    # # cap.set(3, 4096) # At 4096 the parameter is limited to 2304 pixels
    # # cap.set(4, 3072) # At 3072 the parameter is limited to 1536 pixels
    # # cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640) # At 1280 the parameter is limited to 1280 pixels  --> 16/9 format
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1920) # At 1024 the parameter is limited to 720 pixels
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1080) #the parameter is limited to 720 pixels

    # print "CV_CAP_PROP_BRIGHTNESS:", cap.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
    # print "CV_CAP_PROP_CONTRAST:", cap.get(cv2.cv.CV_CAP_PROP_CONTRAST)
    # print "CV_CAP_PROP_FOURCC:", cap.get(cv2.cv.CV_CAP_PROP_FOURCC)
    # print "CV_CAP_PROP_FPS:", cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # print "CV_CAP_PROP_GAIN:", cap.get(cv2.cv.CV_CAP_PROP_GAIN)
    # print "CV_CAP_PROP_EXPOSURE:", cap.get(cv2.cv.CV_CAP_PROP_EXPOSURE)
    # print "CV_CAP_PROP_FRAME_HEIGHT:", cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    # print "CV_CAP_PROP_GAIN:", cap.get(cv2.cv.CV_CAP_PROP_GAIN)
    # print "CV_CAP_PROP_POS_AVI_RATIO:", cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
    # print "CV_CAP_PROP_FORMAT:", cap.get(cv2.cv.CV_CAP_PROP_FORMAT)
    # print "CV_CAP_PROP_FPS:", cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # print "CV_CAP_PROP_CONVERT_RGB:", cap.get(cv2.cv.CV_CAP_PROP_CONVERT_RGB)
    # print "CV_CAP_PROP_FRAME_COUNT:", cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    # print "CV_CAP_PROP_FRAME_WIDTH:", cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    # print "CV_CAP_PROP_MODE:", cap.get(cv2.cv.CV_CAP_PROP_MODE)

    # while(cap.isOpened()):
    #
    #     ret, frame = cap.read()
    #     cv2.imshow('frame', frame)
    #
    #     # if we push "ESC" the program executes
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         print "The input video is stopped"
    #         break
    # cv2.destroyAllWindows()
    # cap.release()
    # Wait until the user hit any key

    # Testing data class 0. Define the testing data set as class 0, since it has not been classified.
    # imgTesting = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed1.jpg", cv2.CV_LOAD_IMAGE_COLOR)

if __name__ == '__main__':
    main()
