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
import pylab                        # required for arrange doing the wx list
import random                       # required to choose random initial weights and shuffle data
from PIL import Image
from planar import BoundingBox      # Required to use https://pythonhosted.org/planar/bbox.html

from FigureDetectionCamera import Perceptron # Here we can get the function from the Python file, called FigureDetectionCamera.py

##########################################
# Classes
##########################################

class ProcessImage(object):

    #The constructor will run each time an object is assigned to this class.
    def __init__(self, img, classStamp):

        # Store the image argument into the public variable img
        self.img = img

        # Set the threshold value to 128, in order to have gray pixels for sprouts and seeds and black pixels for background.
        self.thresholdValue = 128

        #Hardcoded values in HSV which has nice properties to find the sprouts.
        self.minHue = 22
        self.maxHue = 255
        self.minSaturation = 0
        self.maxSaturation = 255
        self.minValue = 147
        self.maxValue = 255

        # The hue min and max range.
        self.lower_hsv = np.array([self.minHue, self.minSaturation, self.minValue], dtype=np.uint8)
        self.upper_hsv = np.array([self.maxHue, self.maxSaturation, self.maxValue], dtype=np.uint8)

        # Some primary image processing
        self.imgGray = self.getGrayImage()
        self.imgHSV = self.getHSV(self.lower_hsv, self.upper_hsv)
        self.imgMorph = self.getOpening(self.imgHSV, 1, 2)          # Remove any noise pixels with erosion first and then dilate after (Opening)
        self.imgMorph = self.getClosing(self.imgMorph, 1, 0)      # However erode again after to get same sprouts size again

        #Define the images that we want to work with.
        self.imgThreshold = self.getThresholdImage(self.thresholdValue)  # Let the background be black and the seeds be gray.
        self.imgSeedAndSprout = self.addImg(self.imgThreshold, self.imgMorph) # Let the background be black,seeds be gray and sprouts be white
        # self.imgSeedAndSprout = self.addImg(self.imgThreshold, self.imgHSV) # Let the background be black,seeds be gray and sprouts be white
        # self.imgSprouts = self.imgMorph.copy() # Define the sprouts as just the morphed HSV image.
        # self.imgSeeds = self.subtractImg(self.imgThreshold.copy(), self.imgSprouts) # Define the seeds as the threshold image without the sprouts

        # contours
        self.contoursThreshold = self.getContours(self.imgThreshold)  # Find the contours of the whole objects, to later do some matching...
        # self.contoursSeeds = self.getContours(self.imgSeeds)          # Find contours of seeds, to later find the center of mass (COM) of the seeds.

        #Center of mass, COM
        # self.centerSeeds = self.getCentroid(self.contoursSeeds, 5000) # Find COM of seeds only, and not the whole object with the sprouts. +8000 pixels is approix the area of the smallest whole seed.
        # For each contour in the contoursThreshold, we run trough all the coordinate and

        # In order to draw on the input image, without messing the original, a copy of the input image is made. This is called imgDrawings
        self.imgDrawings = self.img.copy()

        # The contours taking from imgThreshold contains a lot of noice blobs.
        # Therefore a simple areafilter is checking
        self.contoursThresholdFiltered = self.getContoursFilter(self.contoursThreshold, 5000, 50000)

        # Now for each contour, find out which pixels belong as a sprout pixel and seed pixel
        self.features = self.getFeaturesFromContours(self.imgSeedAndSprout, self.contoursThresholdFiltered, classStamp) # The 100 is not testet to fit the smallest sprout

        # Draw the center of mass, on the copy of the input image.
        # Note: Somehow it seems there is a little offset in the input image, when the sprouts are longer.
        self.drawCentroid(self.imgDrawings, self.features[1], 10, (0, 0, 255))

        # Write the hue_mean, hue_std number of sprout pixels, and the height, and width of the boundingBox around the each sprout
        self.getTextForContours(self.imgDrawings, self.features[0])

        # Each feature is ready to take out for plotting
        print "The features[0] is:", self.features[0] # On the zero'th element, the whole resultList is stored. I.e. [[(x,y), hue_mean, hue_std... classStamped],... [..]]
        print "The features[1] is:", self.features[1] # On the first element, all the center seed coordinates is stored.
        print "The features[2] is:", self.features[2] # On the secound element, all the hue_mean values for each sprout is stored.
        print "The features[3] is:", self.features[3] # On the third element, all the hue_std values for each sprout is stored.
        print "The features[4] is:", self.features[4] # On the fourth element, the number of sprout pixels values for each sprout is stored.
        print "The features[5] is:", self.features[5] # On the firth element, the width values for each sprout is stored.
        print "The features[6] is:", self.features[6] # On the sixth element, the height values for each sprout is stored.
        print "The features[7] is:", self.features[7] # On the seventh element, the width/height ratio values for each sprout is stored.
        print "The features[8] is:", self.features[8] # On the eighth element, the classStamp values for each sprout is stored.


    def getContoursFilter(self, contours, minAreaThreshold, maxAreaThreshold):
        temp_contour = []
        print "Length of contours in imgThreshold is:", len(contours)
        # print "contours looks like this:", contours
        for contour in contours:
            #Get the area of the given contour, in order to check if that contour is actually something useful, like a seed or sprout.
            contour_area = cv2.contourArea(contour, False)

            # If the area is below a given threshold, we skip that contour. It simply had to few pixels to represent an object = seed + sprout
            if (contour_area < minAreaThreshold) or (contour_area > maxAreaThreshold):
                # print "The contour area is", contour_area, "and hence skipped"
                continue
            else:
                temp_contour.append(contour)

        print "Length of contours is now reduced to:", len(temp_contour)
        # print "Now contours looks like this:", temp_contour
        return temp_contour

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
                          element[4],                               # The width in pixels of the boundingBox of the sprout
                          element[5],                               # The height in pixels of the boundingBox of the sprout
                          round(element[6], numberOfDecimals)       # The ratio width/height
                          )

    def drawText(self, img, rows, cols, hue_mean, hue_std, numberOfSproutPixels, width, height, ratio):
        offsetCols = 100
        offsetRows = 70
        cv2.putText(img, "mu:" + str(hue_mean), (rows-offsetCols,                       cols+offsetRows),   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(img, "std:" + str(hue_std), (rows-offsetCols,                       cols+2*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(img, "Pixels:" + str(numberOfSproutPixels), (rows-offsetCols,       cols+3*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(img, "width:" + str(width), (rows-offsetCols,                       cols+4*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(img, "height:" + str(height), (rows-offsetCols,                     cols+5*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(img, "ratio:" + str(ratio), (rows-offsetCols,                       cols+6*offsetRows), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    def convertRGB2Hue(self, r, g, b):
        # http://www.rapidtables.com/convert/color/rgb-to-hsv.htm
        r_temp = float(r)/255
        g_temp = float(g)/255
        b_temp = float(b)/255
        c_max = max(r_temp, g_temp, b_temp)
        c_min = min(r_temp, g_temp, b_temp)
        delta = c_max-c_min
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

        if not sproutPixels:
            print "This contour has no sprout pixels..." # Hence there is no sprout at that given seed, i.e. the hue_mean and hue_std is 0.
            return 0, 0, numberOfSproutPixels

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

    def getSproutAndSeedPixelsForEachContour(self, img, contour):
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

    def drawBoundingBoxSprout(self, sprout):
        if not sprout:
            print "This contour did not have any sprout pixels \n"
            return 0,0

        # http://pythonhosted.org//planar/
        bbox = BoundingBox(sprout)
        poly = bbox.to_polygon()
        # print "The boundingbox is has these corners", poly

        # Lets try to draw this
        p1 = (int(poly[0][1]), int(poly[0][0]))
        p2 = (int(poly[1][1]), int(poly[1][0]))
        p3 = (int(poly[2][1]), int(poly[2][0]))
        p4 = (int(poly[3][1]), int(poly[3][0]))
        cv2.line(self.imgDrawings, p1, p2, (0, 255, 0), 5)
        cv2.line(self.imgDrawings, p2, p3, (0, 255, 0), 5)
        cv2.line(self.imgDrawings, p3, p4, (0, 255, 0), 5)
        cv2.line(self.imgDrawings, p4, p1, (0, 255, 0), 5)

        return bbox.width, bbox.height

    # Note: The argument is: img --> self.imgSeedAndSprout, contours --> self.contoursThreshold, 5000, 50000
    def getFeaturesFromContours(self, img, contours, classStamp):

        # Define the list of elements
        elementList = []

        # Define the list of values for all the contours for each element.
        resultList = []
        seedCenterList = []
        hueMeanList = []
        hueStdList = []
        numberOfSproutPixelsList = []
        widthList = []
        heightList = []
        ratioList = []
        classStampList = []

        #Run through each contour in the contour list (contours) and check each pixel in that contour.
        objectCounter = 0
        seedCounter = 0
        sproutCounter = 0
        print "---------------------------------------------------------------"

        for contour in contours:
            # Now we are at the first contour that contains a lot of pixel coordinates, i.g. (x1,y1), (x2,y2),...,(xn,yn)
            objectCounter = objectCounter + 1

            print "This contour contains", len(contour), "pixels coordinates" # So this is how many pixel sets (x,y) there is in each contour.

            #Then we check each pixel coordinate in this contour
            # and compair it with the same pixel intensity in the input img image.

            # For each contour, the sprout and seed pixels is founded and stored in the sprouts and seeds lists.
            sprout, seed = self.getSproutAndSeedPixelsForEachContour(img, contour)

            # Draw the bounding box for each sprout
            width, height = self.drawBoundingBoxSprout(sprout)
            print "The boundingBoxs width is:", width
            print "The boundingBoxs height is:", height

            if not height:
                ratio = 0
                print "The ratio width/height is:", ratio
            else:
                ratio = float(height/width)
                print "The ratio width/height is:", float(height/width)  # Note: Here I swopped the height and width, since the output was swopped

            # A check for each image how many sprouts, seeds and objects there were.
            if len(sprout) > 0:
                sproutCounter = sproutCounter + 1

            if len(seed) > 0:
                seedCounter = seedCounter + 1

            # Here the COM is calculated out from the pixel that is contained in the seed list.

            # and the COM coordinate is appended to a COM list together with the RGB value for the sprouts.
            # test = np.array(seed, dtype=np.uint8)
            # print self.getCentroidOfSingleContour(test)

            # Append the seed pixels into a temp_array in order to find the COM
            temp_array = []
            temp_array.append(seed)

            center_of_mass = self.getCentroidOfSingleContour(temp_array)
            elementList.append(center_of_mass)

            hue_mean, hue_std, numberOfSproutPixels = self.analyseSprouts(sprout)
            elementList.append(hue_mean)
            elementList.append(hue_std)
            elementList.append(numberOfSproutPixels)
            elementList.append(height)              # Note: Here I swopped the height and width, since the output was swopped
            elementList.append(width)
            elementList.append(ratio)
            elementList.append(classStamp)

            # Finally append the element list into the resultList
            resultList.append(elementList)

            # Append to seedCenter in order to show the COM on the input image
            seedCenterList.append(center_of_mass)

            # Append to all the feature lists in order to plot these feature later on a feature space diagram.
            # However it might end up with a 2D plot with the y-axis as hue_mean and the x-axis as ratio of width/height
            # Maybe we need to have convex hull, but I gues a simple bounding box is enough. I am not using the rotated bbox,
            # since it did not work in OpenCV. The normal boundingbox is from the planar library.
            hueMeanList.append(hue_mean)
            hueStdList.append(hue_std)
            numberOfSproutPixelsList.append(numberOfSproutPixels)
            widthList.append(height)            # Note: Here I swopped the height and width, since the output was swopped
            heightList.append(width)
            ratioList.append(ratio)
            classStampList.append(classStamp)

            # Clear the element list for each contour
            elementList = []

            print "-----------------------Done with that given contour --------------------------------------------"

        print "objectCounter is", objectCounter
        print "seedCounter is", seedCounter
        print "sproutCounter is", sproutCounter

        return resultList, seedCenterList, hueMeanList, hueStdList, numberOfSproutPixelsList, widthList, heightList, ratioList, classStampList

    def showImg(self, nameOfWindow, image, scale):
        imgCopy = image.copy()
        image_show = self.scaleImg(imgCopy, scale)
        cv2.imshow(nameOfWindow, image_show)
        cv2.imwrite("/home/christian/workspace_python/MasterThesis/SeedDetection/writefiles/" + str(nameOfWindow) + ".jpg", image_show)

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

    def getGrayImage(self):
        #Do the grayscale converting
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def getThresholdImage(self, maxValue):
        # Do the thresholding of the image.
        # Somehow we need to return the "ret" together with the image, to be able to show the image...
        ret, img_threshold = cv2.threshold(self.imgGray, self.thresholdValue, maxValue, cv2.THRESH_BINARY)
        return img_threshold

    def getEdge(self):
        img_edges = cv2.Canny(self.imgGray, self.minCannyValue, self.maxCannyValue)
        return img_edges

    def getHSV(self, lower_hsv, upper_hsv):
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img_seg = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        return img_seg

    def getContours(string, binary_img):
        #Copy the image, to avoid manipulating with original
        contour_img = binary_img.copy()

        #Find the contours of the thresholded image
        #Note: See OpenCV doc if needed to change the arguments in findContours.
        # Note: The CHAIN_APPROX_SIMPLE only takes the outer coordinates. I.e. a square has only four coordinates instead
        # of following the edge all around. The CHAIN_APPROX_NONE takes all the pixels.
        # contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
            return

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

    def getMinAreaRect(self, contours):
        list_of_rects = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            list_of_rects.append(rect)
        return list_of_rects

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

    def getOpening(self, img_binary, iterations_erode, iterations_dilate):
        kernel = np.ones((3, 3), np.uint8)
        img_erode = cv2.erode(img_binary, kernel, iterations=iterations_erode)
        img_morph = cv2.dilate(img_erode, kernel, iterations=iterations_dilate)
        return img_morph

    def getClosing(self, img_binary, iterations_erode, iterations_dilate):
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_binary, kernel, iterations=iterations_dilate)
        img_morph = cv2.erode(img_dilate, kernel, iterations=iterations_erode)
        return img_morph

    def nothing(self, x):
        pass

    def addTrackbar(self, name, nameOfWindow, startValue, maxValue):
        cv2.namedWindow(nameOfWindow)
        cv2.createTrackbar(name, nameOfWindow, startValue, maxValue, self.nothing)

    def trackbarListener(self, name, nameOfWindow):
        value = cv2.getTrackbarPos(name, nameOfWindow)
        return value

class PlotFigures():
    def __init__(self, string):
        self.name = string
        self.fig = plt.figure(num=self.name, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        plt.title(self.name)
        self.ax = plt.subplot(111)

    def plotData(self, x, y, string_icon, string_label):
        plt.plot(x, y, string_icon, label=string_label)

        # Shrink current axis by 20%
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # Put a legend to the right of the current axis
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #Set grid on, limit the y axis (not the x yet) and put names on axis
        plt.grid(True)

    def setXlabel(self, string_x):
        plt.xlabel(string_x)

    def setYlabel(self, string_y):
        plt.ylabel(string_y)

    def limit_y(self, min_y, max_y):
        plt.ylim(min_y, max_y)

    def limit_x(self, min_x, max_x):
        plt.xlim(min_x, max_x)

    def plotMean(self, x, y, string_icon):
        plt.plot(np.mean(x), np.mean(y), string_icon, markersize=20)

    def updateFigure(self):
        plt.show(block=False)

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
    ################################################################################################

    # Training data class 1. Define the first testing data set as class 1
    imgTrainingClass1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed1.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # Training data class -1. Define the secound testing data set as class -1
    imgTrainingClassNeg1 = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed2.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # Testing data class 0. Define the testing data set as class 0, since it has not been classified.
    imgTesting = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed1.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    # In order to show an image in HD, the image is way too big to show on a normal size screen.
    # However when calling the object.showImg the image is scaled with the image_ratio parameter.
    # 1 = normal size. 0.5 = half sieze. About 1 and bigger takes a lot of CPU power!!!
    image_ratio = 0.3

    # Do the image processing on the training data class 1
    td1 = ProcessImage(imgTrainingClass1, 1)

    # Do the image processing on the training data class -1.
    tdNeg1 = ProcessImage(imgTrainingClassNeg1, -1)   #Uncomment this one when all the debugging printout is removed.

    # Do the image processing on the testing data class 0
    # testData = ProcessImage(imgTrainingClassNeg1, 0)  #Uncomment this one when all the debugging printout is removed.

    #Draw data
    drawData1 = PlotFigures("Feature space for training data class 1")
    drawData1.plotData(td1.features[2], td1.features[7], "rs", "Class 1")
    drawData1.plotData(tdNeg1.features[2], tdNeg1.features[7], "bs", "Class -1")
    drawData1.setXlabel('hue_mean')
    drawData1.setYlabel('width/height ratio')
    drawData1.updateFigure()


    # td1.showImg("Input image and fitted to display on screen", td1.img, image_ratio)

    # Show the grayscale image
    #td1.showImg("Grayscale image and fitted to display on screen", td1.imgGray, image_ratio)

    # Show the thresholded image. This contains the seeds and the sprouts with a gray level. Background is black
    # td1.showImg("Thresholded image and fitted to display on screen", td1.imgThreshold, image_ratio)

    # Show the segmentated sprouts by using HSV
    # td1.showImg("HSV segmented image and fitted to display on screen", td1.imgHSV, image_ratio)

    # Show the morph on the HSV to get nicer sprouts
    # td1.showImg("HSV segmented image morphed and fitted to display on screen", td1.imgMorph, image_ratio)

    # Show the addition of the two images, thresholde and HSV
    td1.showImg("Added training data class 1 and fitted to display on screen", td1.imgSeedAndSprout, image_ratio)
    tdNeg1.showImg("Added training data class -1 and fitted to display on screen", tdNeg1.imgSeedAndSprout, image_ratio)

    # Show the sprouts image
    #td1.showImg("The sprouts images", td1.imgSprouts, image_ratio)

    # Show te seed image
    #td1.showImg("The seeds images", td1.imgSeeds, image_ratio)

    # Show the input image with indicated center of mass coordinates of each seed.
    td1.showImg("Show trainingdata class 1 with center of mass, boundingBox of sprouts and data in image and let it be fitted to display on screen", td1.imgDrawings, image_ratio)
    tdNeg1.showImg("Show trainingdata class -1 with center of mass, boundingBox of sprouts and data in image and let it be fitted to display on screen", tdNeg1.imgDrawings, image_ratio)


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

if __name__ == '__main__':
    main()
