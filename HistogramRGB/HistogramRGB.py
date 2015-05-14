#!/usr/bin/env python

# From: http://opencv-code.com/tutorials/drawing-histogram-in-python-with-matplotlib/
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_histogram(im, stringTitle, filename):
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
        fig.suptitle(stringTitle, fontsize=22, fontweight='normal')
        size = 18
        font = {'size': size}
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('font', **font)

        fig.add_subplot(311)
        plt.hist(im[...,0].flatten(), 256, range=(0, 254), fc='b')
        plt.xlim(0,255)
        # plt.ylim(0,50000)

        fig.add_subplot(312)
        plt.hist(im[...,1].flatten(), 256, range=(0, 254), fc='g')
        plt.xlim(0,255)
        # plt.ylim(0,50000)

        fig.add_subplot(313)
        plt.hist(im[...,2].flatten(), 256, range=(0, 254), fc='r')
        plt.xlim(0,255)
        # plt.ylim(0,50000)
        plt.savefig(saveImagePath + str(filename) + ".png")

        plt.show(block=False)

if __name__ == '__main__':
    # im = cv2.imread("NGR_godeCropped2.png")
    # im = cv2.imread("NGR_brune.jpg")
    im1 = cv2.imread("NGR_bruneFixed1.jpg")
    im2 = cv2.imread("NGR_for_brune_i_virkeligheden.jpg")
    im3 = cv2.imread("NGR_gode.jpg")
    if not (im1  == None):
        title1 = "RGB histogram yellow/brown sprouts"
        title2 = "RGB histogram too brown sprouts"
        title3 = "RGB histogram white sprouts"
        show_histogram(im1, title1, "yellowAndBrown")
        show_histogram(im2, title2, "tooBrown")
        show_histogram(im3, title3, "white")
        plt.show(block = True)
    else:
        print "No image"
