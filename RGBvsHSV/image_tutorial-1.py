# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import cv2

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

def getRGB(img):
    # Array slicing. OpenCV use BRG, but matplotlib use RGB
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    return r,g,b

def getHSV(img):
    # Array slicing. OpenCV use BRG, but matplotlib use RGB
    # Get the R channel:

    # First convert the rgb image into HSV, using OpenCV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = imgHSV[:,:,0]
    s = imgHSV[:,:,1]
    v = imgHSV[:,:,2]
    return h,s,v

def convertRGB2HSV(r,g,b):
    print "The r value is:", r
    # http://www.rapidtables.com/convert/color/rgb-to-hsv.htm
    r_temp = float(r)/255
    g_temp = float(g)/255
    b_temp = float(b)/255
    c_max = max(r_temp, g_temp, b_temp)
    c_min = min(r_temp, g_temp, b_temp)
    delta = c_max-c_min

    if delta is 0:
        hue = 0
    else:
        if c_max is r_temp:
            hue = 60 * (((g_temp-b_temp)/delta) % 6)
        elif c_max is g_temp:
            hue = 60 * (((b_temp-r_temp)/delta) + 2)
        elif c_max is b_temp:
            hue = 60 * (((r_temp-g_temp)/delta) + 4)
        else:
            print "Debug. Should not get into this else"

    return hue

def main():

    # Reading image created from Gimp
    imgLoop = cv2.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    # roi = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi.jpg')
    # roi = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/4pixelTest.tif')

    # Four pixel image
    # test1 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/4pixelTest.tif')
    # test1 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/4pixelTest2.tif')
    # test1 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_seedOnly.jpg')
    test1 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromDigitalCamera/roi_sproutScaled.tif')
    # test1 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_sproutScaled.tif')

    # Show the image. This shows that the image is mirrowed, since the X-Y coordinate in the image differs.
    # So we show the image, to get everything correct.
    test1 = np.flipud(test1)
    imgplot = plt.imshow(test1)

    # Two pixel image
    # test2 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/2pixel.jpg')
    test2 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromDigitalCamera/roi_seedScaled.tif')
    # test2 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_seedScaled.tif')

    # 1 pixel image
    # test3 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/1pixel.jpg')
    test3 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromDigitalCamera/roi_background.tif')
    # test3 = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_backgroundScaled.tif')

    # roiSeed = mpimg.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_seedOnly.jpg")
    # roiSprout = mpimg.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_sproutOnly.jpg")
    # roiBackground = mpimg.imread("/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/roi_background.jpg")

    # Show the image
    # imgplot = plt.imshow(roiSeed)
    # plt.show()
    # imgplot = plt.imshow(roiSprout)
    # plt.show()
    # imgplot = plt.imshow(roiBackground)
    # plt.show()

    # Plot the 3D plot for RGB
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    for c, m, testImage in [('r', 'o', test1), ('g', 'x', test2), ('b', '^', test3)]:
        r, g, b = getRGB(testImage)
        ax.scatter(r, g, b, c=c, marker=m)

    ax.set_xlabel('R value')
    ax.set_ylabel('G value')
    ax.set_zlabel('B value')

    # Plot the 3D plot for RGB
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')

    r,g,b = getRGB(test1)
    print "For test1 RGB"
    print r
    print "\n"
    print g
    print "\n"
    print b
    print "\n"

    r,g,b = getRGB(test2)
    print "For test2 RGB"
    print r
    print "\n"
    print g
    print "\n"
    print b
    print "\n"

    r,g,b = getRGB(test3)
    print "For test3 RGB"
    print r
    print "\n"
    print g
    print "\n"
    print b
    print "\n"


    h,s,v = getHSV(test1)
    print "For test1 HSV"
    print h
    print "\n"
    print s
    print "\n"
    print v
    print "\n"

    h,s,v = getHSV(test2)
    print "For test2 HSV"
    print h
    print "\n"
    print s
    print "\n"
    print v
    print "\n"

    h,s,v = getHSV(test3)
    print "For test3 HSV"
    print h
    print "\n"
    print s
    print "\n"
    print v
    print "\n"

    for c, m, testImage in [('r', 'o', test1), ('g', 'x', test2), ('b', '^', test3)]:
        h,s,v = getHSV(testImage)
        ax.scatter(h, s, v, c=c, marker=m)
    ax.set_xlabel('H value')
    ax.set_ylabel('S value')
    ax.set_zlabel('V value')

    print "Now we almost at the loop"
    plt.show()

    # In order to close all figures down, click on the roi.jpg image on the screen and push ESC
    # while(1):
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         print "User closed the program..."
    #         break
    #
    # plt.close('all')
    # cv2.destroyAllWindows()

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
