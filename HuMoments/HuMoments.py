#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import classes from component files
from Input import Input
from Preprocessing import Preprocessing
from Segmentation import Segmentation
from PlotFigures import PlotFigures

# Import other libraries
import cv2
import numpy as np

def runStuff(imgInput, saveToImagePath, readFromImagePath, hueMomentIndexX, hueMomentIndexY):
    p = Preprocessing(saveToImagePath, readFromImagePath, imgInput)
    s = Segmentation(saveToImagePath, readFromImagePath, p.imgMorph, imgInput)
    listX = s.getHuMomentsOfAllContours(s.contoursFrontGroundFiltered, hueMomentIndexX)
    listY = s.getHuMomentsOfAllContours(s.contoursFrontGroundFiltered, hueMomentIndexY)

    print "So the listX is:", listX
    print "So the listY is:", listY

    return listX, listY

def main():

    saveToImagePath = "/home/christian/workspace_python/MasterThesis/HuMoments/writefiles/"
    readFromImagePath = "/home/christian/workspace_python/MasterThesis/HuMoments/readfiles/"
    pf = PlotFigures(saveToImagePath)
    i = Input(saveToImagePath, readFromImagePath)

    hueMomentX = 0
    hueMomentY = 1

    listXa, listYa = runStuff(i.trainingData1, saveToImagePath, readFromImagePath, hueMomentX, hueMomentY)
    listXb, listYb = runStuff(i.trainingData2, saveToImagePath, readFromImagePath, hueMomentX, hueMomentY)
    listXc, listYc = runStuff(i.trainingData3, saveToImagePath, readFromImagePath, hueMomentX, hueMomentY)

    pf.plotData(listXa, listYa, "s", "r", "Training data 1", hueMomentX, hueMomentY)
    pf.plotData(listXb, listYb, "s", "g", "Training data 2", hueMomentX, hueMomentY)
    pf.plotData(listXc, listYc, "s", "b", "Training data 3", hueMomentX, hueMomentY)
    pf.updateFigure("HuMoments"+str(hueMomentX)+"vs"+str(hueMomentY))

    # cv2.imshow("Binary image", s.imgDraw)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()