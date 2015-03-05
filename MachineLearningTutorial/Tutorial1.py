# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:54:03 2014

@author: christian
"""

import numpy as np
import mlpy
#D. Albanese, R. Visintainer, S. Merler, S. Riccadonna, G. Jurman, C. Furlanello. mlpy: Machine Learning Python, 2012. arXiv:1202.6548 [bib]
import matplotlib.pyplot as plt # required for plotting
import cv2
import cv

#Load the Iris dataset:
iris = np.loadtxt('iris.csv', delimiter=',')

# x: (observations x attributes) matrix, y: classes (1: setosa, 2: versicolor, 3: virginica)
x, y = iris[:, :4], iris[:, 4].astype(np.int)

#Print what size the x and y has. 
# So x is a 150 times 4 matrix and y is a 150 times 1, or vector
# Try print(x) and print(y) which will reviel the datashet
print(x.shape)
print(y.shape)

#Dimensionality reduction by Principal Component Analysis (PCA)
pca = mlpy.PCA()
pca.learn(x) # learn from data
z = pca.transform(x, k=2) # embed x into the k=2 dimensional subspace
z.shape
print(z.shape)

####
# Important note: Error will show up if the plt.set_cmap(plt.cm.Paired)
# is placed before the plot = plt.scatter(z[:, 0], z[:, 1], c=y)
# http://mlpy.sourceforge.net/docs/3.5/tutorial.html#tutorial-1-iris-dataset
##
fig1 = plt.figure(1)
title = plt.title("PCA on iris dataset")
title = plt.title("PCA on iris dataset")
plot = plt.scatter(z[:, 0], z[:, 1], c=y)
plt.set_cmap(plt.cm.Paired)
labx = plt.xlabel("First component")
laby = plt.ylabel("Second component")
plt.show()

#Learning by Kernel Support Vector Machines (SVMs) on principal components:
linear_svm = mlpy.LibSvm(kernel_type='sigmoid') # new linear SVM instance
linear_svm.learn(z, y) # learn from principal components

#For plotting purposes, we build the grid where we will compute the predictions (zgrid):
xmin, xmax = z[:,0].min()-0.1, z[:,0].max()+0.1
ymin, ymax = z[:,1].min()-0.1, z[:,1].max()+0.1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
zgrid = np.c_[xx.ravel(), yy.ravel()]

#Now we perform the predictions on the grid. The pred() method returns the prediction for each point in zgrid:
yp = linear_svm.pred(zgrid)

#Plot the predictions:
####
# Important note: Error will show up if the plt.set_cmap(plt.cm.Paired)
# is placed before the plt.pcolormesh(xx, yy, yp.reshape(xx.shape))
# http://mlpy.sourceforge.net/docs/3.5/tutorial.html#tutorial-1-iris-dataset
##
fig2 = plt.figure(2)
title = plt.title("SVM (linear kernel) on principal components")
plot1 = plt.pcolormesh(xx, yy, yp.reshape(xx.shape))
plot2 = plt.scatter(z[:, 0], z[:, 1], c=y)
plt.set_cmap(plt.cm.Paired)
labx = plt.xlabel("First component")
laby = plt.ylabel("Second component")
limx = plt.xlim(xmin, xmax)
limy = plt.ylim(ymin, ymax)
plt.show()

#We can try to use different kernels to obtain: