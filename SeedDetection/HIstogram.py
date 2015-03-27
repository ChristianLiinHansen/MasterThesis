#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

# Read data
d1b = np.genfromtxt('workfileClass0Before.txt', delimiter=',')
d2b = np.genfromtxt('workfileClass1Before.txt', delimiter=',')
d3b = np.genfromtxt('workfileClass-1Before.txt', delimiter=',')

d1a = np.genfromtxt('workfileClass0After.txt', delimiter=',')
d2a = np.genfromtxt('workfileClass1After.txt', delimiter=',')
d3a = np.genfromtxt('workfileClass-1After.txt', delimiter=',')


fig = plt.figure(facecolor="white")
ax = fig.add_subplot(111)
ax.set_title('Histogram for each class', fontsize = 20)
plt.hist(d1b, bins=20, histtype='stepfilled', color='b', label='Class 0', alpha = 1)
plt.hist(d2b, bins=20, histtype='stepfilled', color='r', label='Class 1', alpha = 1)
plt.hist(d3b, bins=20, histtype='stepfilled', color='g', label='Class -1', alpha = 1)
plt.ylim(0,10)
plt.xlim(0,2000)
plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
fig.savefig("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section6/Histogram.png" ,dpi=300)


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
