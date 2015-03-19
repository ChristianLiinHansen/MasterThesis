# -*- coding: utf-8 -*-
"""
Created on 18/3-2015
@author: christian
"""

##########################################
# Libraries
##########################################
import matplotlib.pyplot as plt     # required for plotting
import matplotlib.image as mpimg    # required for showimg and reading images with matplotlib
import numpy as np                  # required for calculate i.e mean with np.mean

import cv2                          # required for use OpenCV
import pylab as pl                  # required for arrange doing the wx list
import random                       # required to choose random initial weights and shuffle data

##########################################
# Classes
##########################################

def main():

    # Reading by using matplotlib library. See http://matplotlib.org/users/image_tutorial.html
    # img = cv2.imread(, cv2.CV_LOAD_IMAGE_COLOR)

    # img = mpimg.imread("/home/christian/Dropbox/E14/Master-thesis-doc/images/Improoseed_4_3_2015/images_with_15_cm_from_belt/trainingdata_with_par4/NGR/NGR_optimale.jpg")
    # print "The datatype is:", img.dtype

    img = mpimg.imread('../readfiles/stinkbug.png')
    imgplot = plt.imshow(img)







    # Show the input image
    # cv2.imshow("Input image", img)

    # Do a ROI, so we only look at one object = Sprout + seed with some back ground
    # roi = img[xStart:xEnd, yStart:yEnd]. If no space, it is from either beginning or end.
    # roi = img[490:550, 470:550] # An object
    roi = img[600:, 0:] # The background
    # cv2.imshow("Cropped image", roi)
    cv2.imwrite("/home/christian/workspace_python/MasterThesis/RGBvsHSV/writefiles/roi.jpg", roi)

    # Now the images has been created by using Gimp
    roiSeed = cv2.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_seed.jpg")
    roiSprout = cv2.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_sprout.jpg")
    roiBackground = cv2.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_background2.jpg")


    # Wait until the user hit any key
    cv2.waitKey(0)
    print "User close the program..."
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


    # Extra stuff that might come in handy later...
    # Split the input image into RGB channels. In OpenCV is uses B,R,G. When splitting each channel will be seen as a grayscale, where the highere
    # intensity, the more given color is there in that given pixel.
    # b,g,r = cv2.split(img)

    # If we want to clear out any color in the image, we can do this:
    # img[:,:,0] = 0      # Clear all the blue pixels
    # img[:,:,1] = 0      # Clear all the green pixels
    # cv2.imshow("Input image after manipulating", img)



    # Convert and split the input image to HSV channels
    # imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h,s,v img.split()