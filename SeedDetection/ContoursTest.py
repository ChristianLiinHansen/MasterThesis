# -*- coding: utf-8 -*-

import numpy as np
import cv2

# How to import af class from a file. This imports te TestClass from the Test.py file. The Test.py file lies in the same folder as this script does.
from Test import TestClass

def convertFormatForMinRectArea(listOfPixels):
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

def getClosing(kernelSize, img_binary, iterations_erode, iterations_dilate):
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

def getOpening(kernelSize, img_binary, iterations_erode, iterations_dilate):
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

def getBoxPoints(rect):
    # http://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    p1 = (box[0][0], box[0][1])
    p2 = (box[1][0], box[1][1])
    p3 = (box[2][0], box[2][1])
    p4 = (box[3][0], box[3][1])
    return p1, p2, p3, p4

def drawBoundingBox(p1, p2, p3, p4, imgDraw, boundingBoxColor):
        # Draw the oriente bouningbox
        lineWidth = 1
        cv2.line(imgDraw, p1, p2, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p2, p3, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p3, p4, boundingBoxColor, lineWidth)
        cv2.line(imgDraw, p4, p1, boundingBoxColor, lineWidth)

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

def main():

    # By calling the downfollowing, the constructor for the class TestClass in the file Test.py is callled.
    tc = TestClass()
    tc.testFunction()

    # Read the RGB input image
    img = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    cv2.imshow("Input image", img)

    # Convert RGB to grayscale
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Convert grayscale to binary
    ret,thresh = cv2.threshold(imgray,128,128,0)

    # Perform morph on binary image
    imgMorph = getClosing(3, thresh, 3, 3)
    imgThreshold = imgMorph.copy()          # Copy the thresholded + morph mage befor findContours, since findContours somehow mess the input image.

    # Convert the RGB input to HSV image
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Segment the HSV image by uing inRance.
    lower_hsv = np.array([27, 0, 147], dtype=np.uint8)
    upper_hsv = np.array([128, 255, 255], dtype=np.uint8)
    imgSeg = cv2.inRange(imgHSV, lower_hsv, upper_hsv)
    cv2.imshow("Test of HSV", imgSeg)

    # Perform morph on HSV
    imgMorphSeg = getClosing(3,imgSeg, 3, 3)
    cv2.imshow("Test of HSV morphed closing", imgMorphSeg)

    # Add the HSV together with the threshold image
    addedImage = cv2.add(imgMorphSeg, imgThreshold)
    cv2.imshow("Added image", addedImage)

    # Find the contours in the binary morped image
    contours, hierarchy = cv2.findContours(imgMorph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print "A single contour looks like this", contours[0], contours[1]
    # print "The contours look like this:", contours

    # Draw the contours
    imgDraw = imgMorph.copy()
    imgDraw = cv2.cvtColor(imgDraw,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imgDraw,contours,-1,(0,255,0),1)
    # cv2.imshow("Test of contour", imgDraw)

    # Draw a circle at the given pixel
    # x = contours[0][0][0][0]
    # y = contours[0][0][0][1]
    # cv2.circle(imgDraw, (x, y), 5, (255, 0, 0), -1)
    # cv2.imshow("Drawing a pixel", imgDraw)

    # For each contour, calculate the minAreaRect and draw the boundingbox
    listOfRects = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        listOfRects.append(rect)
        p1, p2, p3, p4 = getBoxPoints(rect)
        drawBoundingBox(p1, p2, p3, p4, imgDraw, (0, 0, 255))

    cv2.imshow("Drawing the boundingboxes", imgDraw)

    # Now try to crop out a single boundinbox.
    # Cropping is img[y1:y2, x1:x2]
    imgDrawCropped = imgDraw[450:600, 450:600]
    cv2.imshow("Show the cropped imgDraw image", imgDrawCropped)

    # Find out how much the boundingbox is rotated. The format is as followed [ ( (xCOM,yCOM), (widht, height), angle ) , .... , ]
    # print "So the listOfRects is:", listOfRects

    # The 3 contour has this information
    print "The 4. contour has this information", listOfRects[3]
    print "Mark this contour with a blue cirlce", listOfRects[3][0]

    # Draw a circle at the given pixel
    x = int(round(listOfRects[3][0][0],0))
    y = int(round(listOfRects[3][0][1],0))

    #Debugging...
    x1, y1, width1, height1 = cv2.boundingRect(contours[3])
    p1a = (x1, y1)
    p2a = (x1+width1, y1)
    p3a = (x1+width1, y1+height1)
    p4a = (x1, y1+height1)
    drawBoundingBox(p1a, p2a, p3a, p4a, imgDraw, (255, 0, 0))
    # Try cropping out the bounding box, and not the oriented one.

    # Cropping is img[y1:y2, x1:x2]
    imgBBcropped = addedImage[y1:y1+height1, x1:x1+width1]
    cv2.imshow("The croppped boundingbox", imgBBcropped)
    cv2.imwrite("/home/christian/workspace_python/MasterThesis/SeedDetection/writefiles/imgBBcropped.jpg", imgBBcropped)


    # Now try to run trough alll
    print "The shape of the cropped boundingbox is", imgBBcropped.shape
    print "The length is:", len(imgBBcropped)

    sprout = []
    seed = []
    background = []

    # The out for loop prints a several list, where each one has 61 elements.

    # rowCounter = y1
    # colCounter = x1

    rowCounter = 0
    colCounter = 0

    for row in imgBBcropped:
        for pixel in row:
            if pixel == 255:
                print "Hey we found a white pixel at this location: (", colCounter, ",", rowCounter, ")"
            colCounter = colCounter + 1
        rowCounter = rowCounter + 1
        colCounter = 0
        # colCounter = x1

                # sprout.append(pixel)
            # elif pixel == 128:
                # seed.append(pixel)
            # else:
                # background.append(pixel)

    # Now the seed can be examinated.
    print "The x,y,width,height is", x1, y1, width1, height1
    cv2.circle(imgDraw, (x1, y1), 5, (255, 0, 0), -1)
    cv2.circle(imgDraw, (x, y), 5, (255, 0, 0), -1)
    cv2.imshow("Drawing a pixel again", imgDraw)

    # # Now get how much this contour is rotated
    # print "The rotation of this contour is:", listOfRects[3][2]
    # angleRotated = listOfRects[3][2]
    # # Now rotate the image
    #
    # # Find out how much the angle should be rotated back, to be aligned as a nice rectangle
    # print "The contour has orientation of ths:", angleRotated
    #
    # # In order to rotate a color image, we should split the colormap into three channles and rotate each channel.
    # # However this adds the cuputation time, so we just convert it to grayscale in order to rotate the color image onces.
    # imgDrawCroppedGray = cv2.cvtColor(imgDrawCropped,cv2.COLOR_BGR2GRAY)

    # # So rotating it +90 degree is a 90 degree counterclockwise direction.
    # rotatedImg = rotateImage(imgDrawCroppedGray, angleRotated+90)
    # cv2.imshow("Show the rotated cropped image", rotatedImg)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
