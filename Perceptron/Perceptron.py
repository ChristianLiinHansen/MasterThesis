# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 19:58:24 2014

@author: christian
"""

# perceptron algorithm to find a straight line separating two datasets.
# uses pylab to draw graf.

import sys
import numpy
import pylab    
import time
def main():
    # run through every datapoint to move the line closer to the correct side of the individual datapoints.
    
    # random initial guess, weights w(x,y), bias b
    # 0 = x *w[0] + y * w[1] + b 
    # OR 
    # y = w[0]/-w[1] * x + b/-w[1]  
    w = [0.0000001, 0.0000001]
    b = 0.0000001

    # scale of each step
    learning_rate = 0.02 #0.02

    # (x, y, class) 
    datapoints = []

    # class: -1
    datapoints.append((0.3, 0.25, -1))
    datapoints.append((0.1, 0.4, -1))
    datapoints.append((0.2, 0.5, -1))
    datapoints.append((0.4, 0.4, -1))
    datapoints.append((0.5, 0.5, -1))
 
    # class: 1
    datapoints.append((0.6, 0.34, 1))
    datapoints.append((0.7, 0.345, 1))
    datapoints.append((0.8, 0.76, 1))
    datapoints.append((0.9, 0.5, 1))
    datapoints.append((0.78, 0.4, 1))
 
    ############################################################################
    # Creating and drawing the graph
    ############################################################################
    # this seperation is solely for drawing purposes, so the sets can have different colours. 
    # containers for points of value: -1
    xNeg1 = []
    yNeg1 = []
    
    # containers for points of value: 1
    x1 = []
    y1 = []
    
    # add datapoints to containers based on value
    for data in datapoints:
        # if value of datapoint = 1
        if data[2] == 1:
            x1.append(data[0])
            y1.append(data[1])
        # if value of datapoint = -1
        else:
            xNeg1.append(data[0])
            yNeg1.append(data[1])

    # create list of uniformly spread x-coordinates, arange(from,to,resolution)
    wx = pylab.arange(-1,1,0.01)
  
    # create list of y-coordinates on the line for drawing purposes.
    wy = (w[0]*wx)/(-w[1]) + (b)/(-w[1])

    # pylab on
    pylab.ion()
    
    # create pylab plot of datapoints and line.
    thePlot = pylab.plot(x1,y1,'g*',xNeg1,yNeg1,'r8',wx,wy,'b-')
    pylab.ylim((-1,1))
    pylab.grid(True)
    pylab.xlabel("x")
    pylab.ylabel("y")
    
    pylab.draw()

    ## print 'drawn'
    time.sleep(5)
    
    ############################################################################
    # Actual perceptron algorithm
    ############################################################################
    counter = 0
    update = True   # true if update has occured, else terminate algorithm
    while update == True:
        counter += 1
        update = False
        # run through all the datapoints 
        for data in datapoints:   
            # projects datapoint onto line, returns current classification of datapoint.
            # class*(x*w(0) + y*w(1) + b) > 0
            if data[0] * w[0] + data[1]*w[1] + b >= 0:
                proj = 1
            else:
                proj = -1
            # if no error, res = o, else error depends on direction of error, either 1 or -1.
            error = (data[2] - proj) / 2
            # only set update-flag when there is an erroneous classification.
            if error != 0: update = True 
            # if class should be pos, but is neg, error=1 vice versa.
            # weights are updated 		
            w[0] += learning_rate * error * data[0]	
            w[1] += learning_rate * error * data[1]
            b += learning_rate * error
        wy = w[0]*wx/-w[1] + b/-w[1]
        thePlot[2].set_ydata(wy)
        pylab.draw()
        print 'x = ', w[0], ' y = ', w[1], ' b = ', b    
    print 'done'
    print 'x = ', w[0], ' y = ', w[1], ' b = ', b    
    print 'counter =', counter

    time.sleep(50)
    
if __name__ == '__main__':
    main()