#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 1/4-2015
@author: Christian Liin Hansen
"""

import numpy as np
import cv2

class Output(object):
    def __init__(self):
        pass

    def convertUV2XYZ(self, uvList0, uvList1, sizeOfImg):
        # print "The first list is:", uvList0
        # print "The secound list is:", uvList1

        # Placeholder for the x,y,z list
        xyzList0 = []
        xyzList1 = []

        # Using the pin hole model here, with the knowledge of the cameras focal length
        # and the distance Z, which is from the camera lense to the surface.
        # It is requied that the optical angle from the cameras is pendicular to the plan where the seeds are spanned.

        # Z in meter from cameras lense surface to the surface of the plan, where the seeds are.
        # z = 0.30
        z = 30

        # Focal length was measured using the /camera_info topic doing the RSD-course
        focal_x = 1194.773485
        focal_y = 1192.665666

        # By using the pin hole model, as
        # x = (u - cols/2)*z / focal_x
        # y = (v - rows/2)*z / focal_y
        imageCol = sizeOfImg[1]
        imageRow = sizeOfImg[0]

        for index in uvList0:
            x0 = (index[0] - imageCol/2) * z / focal_x
            y0 = (index[1] - imageRow/2) * z / focal_y
            temp0 = (x0, y0, z)
            xyzList0.append(temp0)

        for index in uvList1:
            x1 = (index[0] - imageCol/2) * z / focal_x
            y1 = (index[1] - imageRow/2) * z / focal_x
            temp1 = (x1, y1, z)
            xyzList1.append(temp1)
        return xyzList0, xyzList1


