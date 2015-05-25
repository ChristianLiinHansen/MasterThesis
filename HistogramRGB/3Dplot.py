# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import cv2

def removeValues(arr, val):
    arr = arr.flatten()
    arr = arr[arr != val]
    return arr

def getRGB(img):
    # Array slicing. OpenCV use BRG, but matplotlib use RGB. The dtype is float32
    # The slicing is between 0 and 1. Non-normalize by mulityply with 255
    r = img[:,:,0]*255
    g = img[:,:,1]*255
    b = img[:,:,2]*255

    # Remove all the blackpixels for each slice, which value is 0
    r = removeValues(r, 0)
    g = removeValues(g, 0)
    b = removeValues(b, 0)

    return r, g, b

def getHSV(img):
    # Array slicing. OpenCV use BRG, but matplotlib use RGB
    # Get the R channel:

    # First convert the rgb image into HSV, using OpenCV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = imgHSV[:,:,0]
    s = imgHSV[:,:,1]
    v = imgHSV[:,:,2]
    return h,s,v

def main():

    size = 18
    font = {'size': size}
    plt.rc('xtick', labelsize=size)
    plt.rc('ytick', labelsize=size)
    plt.rc('font', **font)

    pathImage = "16_5_2015/"
    whiteSprouts = mpimg.imread(pathImage+"NGR_gode2_medgodfarveEDITED.png")
    yellowBrownSprouts = mpimg.imread(pathImage+"NGR_bruneEDITED.png")

    # Plot the 3D plot for RGB
    sizeOfDataPoints = 40
    fig1 = plt.figure(figsize=(10, 8.21), dpi=100, facecolor='w', edgecolor='k')
    ax = fig1.add_subplot(111, projection='3d')

    fr = open('rValues', 'w')
    fg = open('gValues', 'w')
    fb = open('bValues', 'w')
    # This for loop only iterate 3 times, since we have " for c, m, ....., in [ (stuff1), (stuff2), (stuff3) ]
    for c, m, image, label, c_mean in [('r', 'x', whiteSprouts, 'White sprout', 'r'), ('b', 'x', yellowBrownSprouts, 'Yellow/brown sprout', 'b')]:
        r, g, b = getRGB(image)
        meanR = np.mean(r)
        meanG = np.mean(g)
        meanB = np.mean(b)
        fr.write(str(r))
        fg.write(str(g))
        fb.write(str(b))

        varR = np.var(r)
        varG = np.var(g)
        varB = np.var(b)
        print "Mean RGB value for " + label + " was:", (meanR, meanG, meanB)
        print "Variance RGB for " + label + " was", (varR, varG, varB)
        print "Std RGB for " + label + " was", (np.sqrt(varR), np.sqrt(varG), np.sqrt(varB))
        ax.scatter(r, g, b, s=sizeOfDataPoints, c=c, marker=m, label=label, alpha=0.5)
        ax.scatter(meanR, meanG, meanB, s=20*sizeOfDataPoints, c=c_mean, marker='*', label=label)
    print "done loop"

    ax.set_xlabel('R value')
    ax.set_ylabel('G value')
    ax.set_zlabel('B value')
    # ax.set_xlim(0,255)
    # ax.set_ylim(0,255)
    # ax.set_zlim(0,255)

    # # A hack for getting the legend.
    red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
    # green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
    ax.legend([blue_proxy, red_proxy],['Yellow-brown sprouts', 'White sprouts'], loc = 'upper right')
    plt.show(block = True)
    print "Done showing 3D plot..."
    plt.savefig(pathImage + "3DRGBplotwithMean.png")

if __name__ == '__main__':
    main()


#
#     roi = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi.jpg')
#     roiSeed = mpimg.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_seedOnly.jpg")
#     roiSprout = mpimg.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_sproutOnly.jpg")
#     roiBackground = mpimg.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_background.jpg")
#
#     # imgplot = plt.imshow(roiSeed)
#     # # imgplot = plt.imshow(roiSeed)
#     # plt.show()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     n = 100
#     for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#         xs = randrange(n, 23, 32)
#         ys = randrange(n, 0, 100)
#         zs = randrange(n, zl, zh)
#         ax.scatter(xs, ys, zs, c=c, marker=m)
#
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#
#     plt.show()
#
# if __name__ == '__main__':
#     main()
#

    # print "roi is: \n", roi, "and the shape is:", roi.shape
    # print "The roi contains a R,G,B image. This is extracted, so the red color channel is:", r.shape
    # print "And the red channel contains", r
    # print "And the green channel contains", g
    # print "And the blue channel contains", b


    # Now that we have the red, green and blue image, we can start looking at the intensity of each pixel
    # I.e. within the red channel image, the pixel at location 0,0 is a scalar between 0 - 255.
    # This value is plotted together with the pixelintensity for the green channel image at the same pixel location
    # The same with the blue channel image. The result is a 3D plot in the RGB plot for a single pixel.

    # Split the RGB image into three color images.
    # b,g,r = cv2.split(roiSeed)
    # print "The size of b is:", b.shape
    # imgplot = plt.imshow(b)
    # imgplot = plt.imshow(g)
    # imgplot = plt.imshow(r)
    # plt.show()
