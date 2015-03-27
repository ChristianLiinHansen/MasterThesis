import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

# Nice homepage about manually calculating boxplots,
#http://www.purplemath.com/modules/boxwhisk3.htm
# Read data
d1b = np.genfromtxt('workfileClass0Before.txt', delimiter=',')
d2b = np.genfromtxt('workfileClass1Before.txt', delimiter=',')
d3b = np.genfromtxt('workfileClass-1Before.txt', delimiter=',')

# print "The d1b dataset is:", d1b
d1b_sorted = np.sort(d1b)
print "The order d1b dataset is:", d1b_sorted
print "The length of the db1 dataset is:", len(d1b_sorted)
print "The Q2 (median) is", np.median(d1b_sorted)
print "Removing Q2 form the the dataset, the left is:",





d1a = np.genfromtxt('workfileClass0After.txt', delimiter=',')
d2a = np.genfromtxt('workfileClass1After.txt', delimiter=',')
d3a = np.genfromtxt('workfileClass-1After.txt', delimiter=',')
d1a_sorted = np.sort(d1a)
print "The order d1a dataset is:", d1a_sorted


data = [d1b, d2b, d3b]

fig = plt.figure(facecolor="white")
ax = fig.add_subplot(111)
ax.set_title('Boxplot for each class', fontsize = 20)
ax.set_xlabel('Classes')
ax.set_ylabel('Square pixels')

plt.xticks([1, 2, 3], ["Class 0", "Class 1", "Class -1"])
plt.ylim(0, 2000)
plt.yticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])

plt.boxplot(data, whis=1.0)
plt.scatter([1, 2, 3], [d1b.mean(), d2b.mean(), d3b.mean()], marker='o', c='r', s = 40)
plt.show()
fig.savefig("/home/christian/Dropbox/E14/Master-thesis-doc/images/Section6/Boxplot.png" ,dpi=300)

# # basic plot
# pl.boxplot(data)
#
# pl.xlabel("Classes")
# pl.ylabel("Square pixels")
# pl.ylim(0,2000)
#
# pl.grid()
# pl.show()