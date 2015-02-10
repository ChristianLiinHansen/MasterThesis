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

##########################################
# Classes
##########################################


class ProcessImage(object):

    #The constructor will run each time an object is assigned to this class.
    def __init__(self, img):

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
        self.imgMorph = self.getClosing(self.imgHSV, 2, 2)

        #Define the images that we want to work with.
        self.imgThreshold = self.getThresholdImage(self.thresholdValue)  # Let the background be black and the seeds be gray.
        self.imgSeedAndSprout = self.addImg(self.imgThreshold, self.imgHSV) # Let the background be black,seeds be gray and sprouts be white
        self.imgSprouts = self.imgMorph.copy() # Define the sprouts as just the morphed HSV image.
        self.imgSeeds = self.subtractImg(self.imgThreshold.copy(), self.imgSprouts) # Define the seeds as the threshold image without the sprouts

        # contours
        self.contoursThreshold = self.getContours(self.imgThreshold)  # Find the contours of the whole objects, to later do some matching...
        self.contoursSeeds = self.getContours(self.imgSeeds)          # Find contours of seeds, to later find the center of mass (COM) of the seeds.

        #Center of mass, COM
        self.centerSeeds = self.getCentroid(self.contoursSeeds, 5000) # Find COM of seeds only, and not the whole object with the sprouts. +8000 pixels is approix the area of the smallest whole seed.

        # For each contour in the contoursThreshold, we run trough all the coordinate and
        self.sproutAndSeedPixels = self.getSproutAndSeedPixels(self.imgSeedAndSprout, self.contoursThreshold, 5000) # The 100 is not testet to fit the smallest sprout


        self.imgWithContours = self.img.copy()
        self.drawCentroid(self.imgWithContours, self.centerSeeds, 10, (0, 0, 255))

        self.rects = self.getMinAreaRect(self.contoursSeeds)
        # self.box = cv2.cv.BoxPoints(self.rects)
        # self.box = np.int0(self.box)

    def getSproutAndSeedPixels(self, img, contours, areaThreshold):
        #cv2.imshow("Test tat img, really is an image", img)
        print "Now we are inside getSproutAndSeedPixels"
        print "The length of the contours is: ", len(contours)
        # List of centers
        seeds = []
        sprouts = []

        print "The shape of the image is: ", img.shape # rols x cols
        #print "The center seed is: ", self.centerSeeds[0][0], self.centerSeeds[0][1]

        # x = self.centerSeeds[0][0]
        # y = self.centerSeeds[0][1]
        # test = img[x,y]
        #
        # print "The intensity value is: ", test

        # Run through each contour in the contour list (contours) and check each pixel in that contour.
        objectCounter = 0
        seedCounter = 0
        sproutCounter = 0
        print "---------------------------------------------------------------"
        for contour in contours:

            # Now we are at the first contour that contains a lot of pixel coordinates, i.g. (x1,y1), (x2,y2),...,(xn,yn)

            #Get the area of the given contour, in order to check if that contour is actually something useful, like a seed or sprout.
            contour_area = cv2.contourArea(contour, False)

            # If the area is below a given threshold, we skip that contour. It simply had to few pixels to represent an object = seed + sprout
            if contour_area < areaThreshold:
                continue

            # Now we examine a contour that is bigger then the areaTHreshold, i.e it is not any noise blobs.
            objectCounter = objectCounter + 1

            print "This contour contains", len(contour), "pixels coordinates" # So this is how many pixel sets (x,y) there is in each contour.

            #Now this contours is above a given acceptable area. Then we check each pixel coordinate in this contour
            # and compair it with the same pixel intensity in the input img image.
            #element = 0

            for pixel in contour:   # Example:
                #element = element + 1
                #print "We are at ", element, "out of ", len(contour)

                #print pixel         # [[3116 2195 ] ... ]
                #print pixel[0]      # [3116 2195]
                #print pixel[0][0]   # 3116  --> Since the shape of the image is a 2195 x 3648, the 3648 is the x, or colums.
                #print pixel[0][1]   # 2195   --> This is the y or rows.

                # But when we have to get the intensity value out, the index of an pixel is called like this:
                # img[rows,cols], so we have to swop, since the (x,y) coordinate in OpenCV read image is (cols, rows)
                # Therefore we swop, in order to make it right.
                row = pixel[0][1]
                col = pixel[0][0]

                # It does not make sence to say pixel[1][0], since we examinate only one pixel at a time, which is on element 0.
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

                #Now compair the intensity value of the input image, (argument in this function), at the given pixelcoordinate
                #which is defined as pixel in the for loop.

            print "The final list with sprouts pixel for this given contour contains", sprouts
            print "And the size of the sprouts list was ", len(sprouts)
            print "The final list with seed pixel for this given contour contains", seeds
            print "And the size of the seed list was ", len(seeds)

            # A check for each image how many sprouts, seeds and objects there were.
            if len(sprouts) > 0:
                sproutCounter = sproutCounter + 1

            if len(seeds) > 0:
                seedCounter = seedCounter + 1

            # Here the COM is calculated out from the pixel that is contained in the seed list.
            # and the COM coordinate is appended to a COM list together with the RGB value for the sprouts.


            

            #Clear the lists of sprouts and seeds pixels, in order to be ready for the next image
            sprouts = []
            seeds = []

            print "-----------------------Done with that given contour --------------------------------------------"

        print "objectCounter is", objectCounter
        print "seedCounter is", seedCounter
        print "sproutCounter is", sproutCounter
        return seeds, sprouts

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
        contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Return the contours. We dont want to use the hierarchy yet
        # However the hierarchy is usefull the detect contours inside a contour or dont detect it.
        # That is what hierarchy keeps track of. (Children and parents)
        return contours

    def getCentroid(self, contours, areaThreshold):
        # List of centers
        centers = []

        #Run through all the contours
        for contour in contours:

            #Get the area of each contour
            contour_area = cv2.contourArea(contour, False)

            if contour_area < areaThreshold:
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


# Test function

def testFunction():
    a = 2
    b = 2
    c = a + b
    return a, b, c

def main():

    # Loading the image.

    #This image, seed1, is an ideal image taking with flash from a high resolution camera. Using images from movie, will not be as nice
    # However I will start implementing the classification system based on a the nice image, and later we can focus on getting better images.

    # Using PIL library to see the size of the image below.
    # im = Image.open("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed1.jpg")
    # print "size is: ", im.size

    #Using the OpenCV standard way of loading an image
    input_image = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed1.jpg", cv2.CV_LOAD_IMAGE_COLOR)

    #print input_image[43,34]

    #input_image = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/seed10.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    #input_image = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/FromVideo.png", cv2.CV_LOAD_IMAGE_COLOR)

    #This image is from the video, where no flash has been used. This results in less quality.
    #input_image = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/seeds/TestingQuality1.png", cv2.CV_LOAD_IMAGE_COLOR)
    # The next step is to interface a camera into the code.
    # And then each frame from the camera will be treated as a input_image

    # Do the image processing on that given image and create the object "image"
    imgObj = ProcessImage(input_image)

    # Show the image. 1 = normal size. 0.5 = half size. About 1 and bigger takes a lot of CPU power!!! So dont go there.
    image_ratio = 0.3
    #imgObj.showImg("Input image and fitted to display on screen", imgObj.img, image_ratio)

    # Grayscale
    #imgObj.showImg("Grayscale image and fitted to display on screen", imgObj.imgGray, image_ratio)

    # Threshold the image. This contains the seeds and the sprouts with a gray level.
    imgObj.showImg("Thresholded image and fitted to display on screen", imgObj.imgThreshold, image_ratio)

    # Segment the image with HSV
    #imgObj.showImg("HSV segmented image and fitted to display on screen", imgObj.imgHSV, image_ratio)

    # Do a little morph on the HSV to get nicer sprouts
    #imgObj.showImg("HSV segmented image morphed and fitted to display on screen", imgObj.imgMorph, image_ratio)

    # Add the two images, grayscale and HSV
    imgObj.showImg("Added images and fitted to display on screen", imgObj.imgSeedAndSprout, image_ratio)
    #imgObj.showImg("The sprouts images", imgObj.imgSprouts, image_ratio)
    #imgObj.showImg("The seeds images", imgObj.imgSeeds, image_ratio)

    # Add the two images, grayscale and HSV
    imgObj.showImg("Show contours in image and let it be fitted to display on screen", imgObj.imgWithContours, image_ratio)

    # Get the list of contours with
    contoursThreshold = imgObj.contoursThreshold

    contoursSeed = imgObj.contoursSeeds
    # Get the list of data with center of mass coordinates of the seeds in the image
    centerSeeds = imgObj.centerSeeds

    result = imgObj.sproutAndSeedPixels
    # print result

    # print "The centerSeeds size is: ", len(centerSeeds)
    # print centerSeeds
    # print centerSeeds[0]

    #print "The contoursThreshold size is: ", len(contoursThreshold)
    #print "The contoursSeed size is: ", len(contoursSeed)
    # print "The contours contains: "
    # print contoursThreshold[1]

    # print "Result of test function is: ", testFunction()
    #
    # test = testFunction()
    # print test[2]

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
