# -*- coding: utf-8 -*-
"""
Support vector machines
@author: Christian Liin Hansen
@date: 23/03-2015
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#example-svm-plot-iris-py
def main():

    # Load training data from the the sklearn library datasets.
    iris = datasets.load_iris()
    # print iris

    # We only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
    X = iris.data[:, :2]

    # Store the "target" from the dataset iris into y
    y = iris.target

    # print "The X is:", X
    # print "The y is:", y

    # Step size in the mesh
    h = 0.02

    # We create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter

    print "The length of y is:", len(y)
    print "The length of X is:", len(X)
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Title for the plots
    titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        print "Z is:", Z

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        print "Now Z is:", Z

        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()

    # print "xx is:", xx
    # print "yy is:", yy
    # print "Z is:", Z
    #
    # print "And the length of xx is:", len(xx)
    # print "And the length of yy is:", len(yy)
    # print "And the length of Z is:", len(Z)
    #
    # file = open('workfile.txt', 'w')
    # file.write(str(xx))

if __name__ == '__main__':
    main()