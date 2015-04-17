#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 17/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

class PlotFigures(object):
    def __init__(self, titleName, fileName):
        self.fileName = fileName
        self.size = 18
        font = {'size': self.size}
        plt.rc('xtick', labelsize=self.size)
        plt.rc('ytick', labelsize=self.size)
        plt.rc('font', **font)

        # self.fig = plt.figure(num=titleName, figsize=(10, 8.21), dpi=300, facecolor='w', edgecolor='k')
        self.fig = plt.figure(num=titleName, figsize=(10, 8.21), dpi=100, facecolor='w', edgecolor='k')
        plt.title(titleName)
        self.ax = plt.subplot(111)

    def plotData(self, x, y, string_icon, string_label):
        plt.plot(x, y, string_icon, label=string_label, markersize=self.size/2)

        # Shrink current axis by 20%
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # Put a legend to the right of the current axis
        # self.ax.legend(loc='center left', bbox_to_anchor=(0.8, 1))
        self.ax.legend(loc='lower right')

        #Set grid on, limit the y axis (not the x yet) and put names on axis
        plt.grid(True)

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
        plt.show(block=True)   # It is very big with 300 dpi
        self.saveFigure()

    def saveFigure(self):
        # plt.annotate('Removed datapoint', xy=(0.33, 0.43), xytext=(0.6, 0.5), arrowprops=dict(facecolor='black', shrink=0.005))
        plt.savefig("/home/christian/workspace_python/MasterThesis/FinalProject/writefiles/" + str(self.fileName) + ".jpg")
