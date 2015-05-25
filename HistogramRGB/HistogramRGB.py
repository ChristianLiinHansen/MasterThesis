#!/usr/bin/env python

# From: http://opencv-code.com/tutorials/drawing-histogram-in-python-with-matplotlib/
import cv2
import numpy as np
import matplotlib.pyplot as plt

def removeValues(arr, val):
    arr = arr.flatten()
    arr = arr[arr != val]
    return arr

def show_histogram(im, filename):
    """ Function to display image histogram.
        Supports single and three channel images. """

    saveImagePath = "/home/christian/workspace_python/MasterThesis/HistogramRGB/Test3/"

    if im.ndim == 2:
        # Input image is single channel
        plt.hist(im.flatten(), 256, range=(0, 250), fc='k')
        plt.show()

    elif im.ndim == 3:
        # Input image is three channels
        fig = plt.figure(figsize=(10, 8.21), dpi=100, facecolor='w', edgecolor='k')
        size = 18
        font = {'size': size}
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('font', **font)

        fig.add_subplot(311)
        fig.tight_layout()
        plt.hist(removeValues(im[...,0].flatten(), 0), 256, range=(0, 255), fc='b')
        plt.xlim(0,255)
        plt.ylim(0,30)
        plt.xlabel("Bins")
        plt.ylabel("Frequency")

        fig.add_subplot(312)
        fig.tight_layout()
        plt.hist(removeValues(im[...,1].flatten(), 0), 256, range=(0, 255), fc='g')
        plt.xlim(0,255)
        plt.ylim(0,60)
        plt.xlabel("Bins")
        plt.ylabel("Frequency")

        fig.add_subplot(313)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1)
        plt.hist(removeValues(im[...,2].flatten(), 0), 256, range=(0, 255), fc='r')
        plt.xlim(0,255)
        plt.ylim(0,200)
        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.savefig(saveImagePath + str(filename) + ".png")

        plt.show(block=True)

if __name__ == '__main__':
    im = cv2.imread("16_5_2015/NGR_gode2_medgodfarveEDITED.png")
    # im = cv2.imread("16_5_2015/NGR_bruneEDITED.png")
    if not (im  == None):
        show_histogram(im, "white")

        while True:
            cv2.imshow("Show the im1 image", im)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                print "User closed down the program..."
                break
    else:
        print "No image"
