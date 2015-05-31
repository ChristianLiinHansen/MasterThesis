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
    def __init__(self, saveImagePath):
        self.saveImagePath = saveImagePath

    def setTitle(self, title):
        plt.title(title)

    def plotData(self, x, y, string_icon, color, string_label, hueMomentX, hueMomentY):
        plt.plot(x, y, string_icon, c=color, label=string_label)

        # Plot legend.
        plt.legend(loc="lower right")

        #Set grid on, limit the y axis (not the x yet) and put names on axis
        plt.grid(True)

        # Plot X label
        plt.xlabel("HuMoment "+str(hueMomentX))
        plt.ylabel("HuMoment "+str(hueMomentY))

    def limit_y(self, min_y, max_y):
        plt.ylim(min_y, max_y)

    def limit_x(self, min_x, max_x):
        plt.xlim(min_x, max_x)

    def plotMean(self, x, y, string_icon):
        plt.plot(np.mean(x), np.mean(y), string_icon, markersize=20)

    def updateFigure(self, fileName):
        plt.show(block=False)   # It is very big with 300 dpi
        plt.savefig(self.saveImagePath + str(fileName) + ".png")

    def clearFigure(self):
        plt.clf()
