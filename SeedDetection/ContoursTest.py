# -*- coding: utf-8 -*-

import numpy as np
from psutil._common import constant
import cv2

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

def getBoxPoints(rect):
    # http://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    p1 = (box[0][0], box[0][1])
    p2 = (box[1][0], box[1][1])
    p3 = (box[2][0], box[2][1])
    p4 = (box[3][0], box[3][1])
    return p1, p2, p3, p4

def drawBoundingBox(p1, p2, p3, p4, imgDraw):
        # Draw the oriente bouningbox
        lineWidth = 1
        boundingBoxColor = (0, 0, 255)
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

    img = cv2.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimale.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)

    imgMorph = getClosing(3,thresh, 3, 3)
    imgThreshold = imgMorph.copy()          # Copy the thresholded + morph mage befor findContours, since findContours somehow mess the input image.
    contours, hierarchy = cv2.findContours(imgMorph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print "A single contour looks like this", contours[0], contours[1]
    print "The contours look like this:", contours

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
        drawBoundingBox(p1, p2, p3, p4, imgDraw)

    # cv2.imshow("Drawing the boundingboxes", imgDraw)

    # Now try to crop out a single boundinbox.
    imgDrawCropped = imgDraw[200:400, 100:300]
    cv2.imshow("Show the cropped imgDraw image", imgDrawCropped)

    # Find out how much the boundingbox is rotated. The format is as followed [ ( (xCOM,yCOM), (widht, height), angle ) , .... , ]
    # print "So the listOfRects is:", listOfRects

    # The 3 contour has this information
    print "The 4. contour has this information", listOfRects[3]
    print "Mark this contour with a blue cirlce", listOfRects[3][0]

    # Draw a circle at the given pixel
    x = int(round(listOfRects[3][0][0],0))
    y = int(round(listOfRects[3][0][1],0))

    cv2.circle(imgDraw, (x, y), 5, (255, 0, 0), -1)
    # cv2.imshow("Drawing a pixel again", imgDraw)

    # Now get how much this contour is rotated
    print "The rotation of this contour is:", listOfRects[3][2]
    angleRotated = listOfRects[3][2]
    # Now rotate the image

    # Find out how much the angle should be rotated back, to be aligned as a nice rectangle
    print "The contour has orientation of ths:", angleRotated

    # In order to rotate a color image, we should split the colormap into three channles and rotate each channel.
    # However this adds the cuputation time, so we just convert it to grayscale in order to rotate the color image onces.
    imgDrawCroppedGray = cv2.cvtColor(imgDrawCropped,cv2.COLOR_BGR2GRAY)

    # So rotating it +90 degree is a 90 degree counterclockwise direction.
    rotatedImg = rotateImage(imgDrawCroppedGray, -30)
    cv2.imshow("Show the rotated cropped image", rotatedImg)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
