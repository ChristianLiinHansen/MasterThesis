#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 17/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import *

class PlotFigures(object):
    def __init__(self, figNumb, figTitle, lowerTitle, saveImagePath):
        self.saveImagePath = saveImagePath
        self.figTitle = figTitle
        self.lowerTitle = lowerTitle

        # self.size = 18
        # font = {'size': self.size}
        # plt.rc('xtick', labelsize=self.size)
        # plt.rc('ytick', labelsize=self.size)
        # plt.rc('font', **font)
        # plt.figure(figNumb, figsize=(10, 8.21), dpi=100, facecolor='w', edgecolor='k')
        # plt.subplot(111)

        self.size = 18
        font = {'size': self.size}
        plt.rc('xtick', labelsize=self.size)
        plt.rc('ytick', labelsize=self.size)
        plt.rc('font', **font)

        self.fig = plt.figure(figNumb, figsize=(10, 8.21), dpi=100, facecolor='w', edgecolor='k')
        self.fig.suptitle(self.figTitle, fontsize=22, fontweight='normal')
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(top=0.90)
        self.ax.set_title(self.lowerTitle)

        # ax.set_xlabel('xlabel')
        # ax.set_ylabel('ylabel')
        # ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

        # plt.subplot(211)
        # plt.plot([1,2,3], label="test1")
        # plt.plot([3,2,1], label="test2")
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #    ncol=2, mode="expand", borderaxespad=0.)

    def setTitle(self, title):
        plt.title(title)

    def plotContourf(self, xx, yy, Z):
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.spectral, alpha=0.8)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.8)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.jet, alpha=0.8)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.prism, alpha=0.8)
        # plt.contourf(xx, yy, Z, cmap=my_cmap, alpha=1)

    def plotData(self, x, y, string_icon, string_label):
        plt.plot(x, y, string_icon, label=string_label, markersize=self.size/2)

        # Shrink current axis by 20%
        # box = self.ax.get_position()
        # self.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # Put a legend to the right of the current axis
        # self.ax.legend(loc='center left', bbox_to_anchor=(0.8, 1))
        # self.ax.legend(loc='lower right')

        #Set grid on, limit the y axis (not the x yet) and put names on axis
        plt.grid(True)

    def addLegend(self):
        plt.legend(loc= "lower right")

    def setXlabel(self, string_x):
        plt.xlabel(string_x, fontsize=self.size)

    def setYlabel(self, string_y):
        plt.ylabel(string_y, fontsize=self.size)

    def limit_y(self, min_y, max_y):
        plt.ylim(min_y, max_y)

    def limit_x(self, min_x, max_x):
        plt.xlim(min_x, max_x)

    def plotMean(self, x, y, string_icon):
        plt.plot(np.mean(x), np.mean(y), string_icon, markersize=20)

    def updateFigure(self):
        plt.show(block=False)   # It is very big with 300 dpi
        draw()
        # self.saveFigure()

    def clearFigure(self):
        plt.clf()

    def saveFigure(self, fileName):
        # plt.annotate('Removed datapoint', xy=(0.33, 0.43), xytext=(0.6, 0.5), arrowprops=dict(facecolor='black', shrink=0.005))
        plt.savefig(self.saveImagePath + str(fileName) + ".png")
