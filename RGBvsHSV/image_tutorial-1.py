# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
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

def main():

    roi = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_sprout.tif')

    # NOTE:. Using flipud,(flip-updown) since image is mirrowed,

    # Reading image created from Gimp - Testing
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/imgTest/1REDpixel.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/imgTest/4pixelTest2.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/imgTest/roi_seedOnly.jpg')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromDigitalCamera/roi_sproutScaled.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_sproutScaled.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_sprout.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_sproutScaled.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_sproutScaled2.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_sprout2.tif')
    # sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_sprout22.png')

    # Final sprout image
    sprout = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_sprout22.tif')
    sprout = np.flipud(sprout)

    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/imgTest/1GREENpixel.tif')
    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromDigitalCamera/roi_seedScaled.tif')
    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_seedScaled.tif')
    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_seed.tif')
    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_seedScaled.tif')
    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_seedScaled2.tif')
    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_seed2.tif')
    # seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_seed22.png')

    # Final seed image
    seed = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_seed22.tif')
    seed = np.flipud(seed)

    # background = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/imgTest/1BLUEpixel.tif')
    # background = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromDigitalCamera/roi_background.tif')
    # background = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_backgroundScaled.tif')
    # background = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/NGR_background.tif')

    # Final background image
    background = mpimg.imread('/home/christian/workspace_python/MasterThesis/RGBvsHSV/readfiles/ImgFromWebcamera/roi_backgroundScaled.tif')
    background = np.flipud(background)

    # Plot the 3D plot for RGB
    sizeOfDataPoints = 40
    markers = 'x'
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.set_title('3D plot for RGB map', fontsize = 20)

    # Now trying to cut down all pixels, that contains

    # This for loop only iterate 3 times, since we have " for c, m, ....., in [ (stuff1), (stuff2), (stuff3) ]
    for c, m, image, label in [('r', markers, sprout, 'Sprout'), ('g', markers, seed, 'Seed'), ('b', markers, background, 'Background')]:
        r, g, b = getRGB(image)
        ax.scatter(r, g, b, s=sizeOfDataPoints, c=c, marker=m, label=label)
    print "done loop"


    ax.set_xlabel('R value')
    ax.set_ylabel('G value')
    ax.set_zlabel('B value')
    ax.set_xlim(0,255)
    ax.set_ylim(0,255)
    ax.set_zlim(0,255)

    # # A hack for getting the legend.
    red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
    green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
    ax.legend([red_proxy,green_proxy, blue_proxy],['Sprout', 'Seed', 'Background'], loc = 'upper right')

    # Plot the 3D plot for HSV
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_title('3D plot for HSV map', fontsize = 20)

    print "Now for the HSV"
    for c, m, image in [('r', markers, sprout), ('g', markers, seed), ('b', markers, background)]:
        h,s,v = getHSV(image)
        print h, s, v
        ax.scatter(h, s, v, s=sizeOfDataPoints, c=c, marker=m)

    ax.set_xlabel('H value')
    ax.set_ylabel('S value')
    ax.set_zlabel('V value')
    ax.set_xlim(0,180)  # The hue range is setted to be from 0 til 180
    ax.set_ylim(0,255)
    ax.set_zlim(0,255)

    # A hack for getting the legend.
    red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
    green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
    ax.legend([red_proxy,green_proxy, blue_proxy],['Sprout','Seed', 'Background'], loc = 'upper right')


    plt.show(block = True)

    # WARNING!. Uncomment only if I want to update figures directly into the documentation.
    # fig1.savefig("/home/christian/Dropbox/E14/Master-thesis-doc/images/ImgFromWebcamera/Resultplots/RGBplot.png" ,dpi=300)
    # fig2.savefig("/home/christian/Dropbox/E14/Master-thesis-doc/images/ImgFromWebcamera/Resultplots/HSVplot.png" ,dpi=300)

    # # In order to close all figures down, click on the roi.jpg image on the screen and push ESC
    # cv2.imshow("Click on this image to close down the program", roi)
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
