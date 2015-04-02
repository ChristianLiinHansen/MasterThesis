#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import classes from component files
import cv2

from Input import Input
from Preprocessing import Preprocessing
from Segmentation import Segmentation
from Classification import Classification
from Output import Output

def main():

    # Initialize the Input component with cameraIndex = 0 (webcamera inbuilt in PC)
    i = Input(0)

    # Initialize the Preprocessing component
    p = Preprocessing()

    # Initialize the Segmentation component
    s = Segmentation()

    # Initialize the Classification component
    c = Classification()

    # Initialize the Output component
    o = Output()

    while i.cameraIsOpen:

        # Input from webcamera
        imgInput = i.getImg()

        # Input from still image
        


        cv2.imshow("Streaming from camera", imgInput)

        # If the user push "ESC" the program close down.
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            i.closeDown()
            break

if __name__ == '__main__':
    main()