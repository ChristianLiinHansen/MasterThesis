#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# Below is code that generated the original histogram, which Henrik did not like since class0, class1 and class2
# did do occlusion for each other. It was not possible to see how the distribution of class0 was, if class1 distribution
# was in front of distribution 0.
# Therefore a histogram of just class0 is created, since it contains a mix of class1 and class -1. This code is below the outcommented code.
# See codeline 38.

# Read data
# d1b = np.genfromtxt('workfileClass0Before.txt', delimiter=',')
# d2b = np.genfromtxt('workfileClass1Before.txt', delimiter=',')
# d3b = np.genfromtxt('workfileClass-1Before.txt', delimiter=',')
#
# d1a = np.genfromtxt('workfileClass0After.txt', delimiter=',')
# d2a = np.genfromtxt('workfileClass1After.txt', delimiter=',')
# d3a = np.genfromtxt('workfileClass-1After.txt', delimiter=',')
#
#
# fig = plt.figure(facecolor="white")
# ax = fig.add_subplot(111)
# ax.set_title('Histogram for each class', fontsize = 20)
# plt.hist(d1b, bins=20, histtype='stepfilled', color='b', label='Class 0', alpha = 1)
# plt.hist(d2b, bins=20, histtype='stepfilled', color='r', label='Class 1', alpha = 1)
# plt.hist(d3b, bins=20, histtype='stepfilled', color='g', label='Class -1', alpha = 1)
# plt.ylim(0,10)
# plt.xlim(0,2000)
# plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()
# fig.savefig("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section6/Histogram.png" ,dpi=300)


# Ok, so this code below is for creating a new histogram, which only shows the histogram of class0.
# Reason for doing this is described in the beginning of this Python file.
# Read data
# d1a or d1b? a = after filering, b = before filtering. It is just to verify that the histogram do not contain more contourareas below 200
# after the contourFiltering funktion was applied. However then the axis must change in order to have the same histogram. This has not been done,
# since there is not going to be two histogram, where the only idea is to show that now there is no contours with area of 200 and below in the image.
# Or, said in an other way. There will still be contours with area less than 200, but when we go through the Pythpn scrip,
# we just ignor/skip, make a "continue" in the for loop of contours, if the area is below 200.
d1b = np.genfromtxt('workfileClass0Before.txt', delimiter=',')
d1a = np.genfromtxt('workfileClass0After.txt', delimiter=',')

fig = plt.figure(facecolor="white")
ax = fig.add_subplot(111)
counts, bins, patches = ax.hist(d1b, bins=10, facecolor='blue', label='Contours', edgecolor='gray')

print "So the legnth of d1b is", len(d1b)

# http://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place...
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

# Label the raw counts and the percentages below the x-axis...
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for count, x in zip(counts, bin_centers):
    # Label the raw counts
    ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')

    # Label the percentages
    percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -52), textcoords='offset points', va='top', ha='center')


# Give ourselves some more room at the bottom of the plot
plt.subplots_adjust(bottom=0.16)
plt.ylim(0,20)
plt.grid()
ax.set_axisbelow(True)
plt.xlabel("Contour area")
plt.ylabel("Frequency")
ax.set_title('Histogram for testing data', fontsize = 20)
plt.legend()
plt.show(fig.savefig("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section6/HistogramClass0.png" ,dpi=300))

# ax.set_title('Histogram for test image', fontsize = 20)
# plt.xlim(0,2000)
# plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
# plt.xlabel("Contour area")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()
# fig.savefig("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section6/HistogramClass0.png" ,dpi=300)
#

# # and exercise the weights option by arbitrarily giving the first half
# # of each series only half the weight of the others:
#
# w0b = np.ones_like(d1b)
# w0b[:len(d1b)/2] = 0.5
# w1b = np.ones_like(d2b)
# w1b[:len(d1b)/2] = 0.5
# w2b = np.ones_like(d3b)
# w2b[:len(d1b)/2] = 0.5
#
# fig1 = P.figure()
# n, bins, patches = P.hist( [d1b,d2b,d3b], 10, weights=[w0b, w1b, w2b], histtype='bar')
#
# # Plot the next figure
# w0a = np.ones_like(d1a)
# w0a[:len(d1a)/2] = 0.5
# w1a = np.ones_like(d2a)
# w1a[:len(d1a)/2] = 0.5
# w2a = np.ones_like(d3a)
# w2a[:len(d1a)/2] = 0.5
# fig2 = P.figure()
# n, bins, patches = P.hist( [d1a,d2a,d3a], 10, weights=[w0a, w1a, w2a], histtype='bar')
# P.show()
#
# #
# # The hist() function now has a lot more options
# #
#
# #
# # first create a single histogram
# #
# fig1 = plt.figure()
# hist1, bins = np.histogram(d1b, bins=50)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist1, align='center', width=width)
# plt.ylim(0, 10)
# plt.show(block = False)
#
# fig2 = plt.figure()
# hist2, bins = np.histogram(d2b, bins=10)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist2, align='center', width=width)
# plt.ylim(0, 10)
# plt.show(block = False)
#
# fig3 = plt.figure()
# hist3, bins = np.histogram(d3b, bins=10)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist3, align='center', width=width)
# plt.ylim(0, 10)
# plt.show()
