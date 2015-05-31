import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Plot3D(object):

    def __init__(self, saveToImagePath, readFromImagePath):
        self.saveToImagePath = saveToImagePath
        self.readFromImagePath = readFromImagePath
        self.fig = plt.figure(figsize=(10, 8.21), dpi=100, facecolor='w', edgecolor='k')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.suptitle("3D RGB plot of background, seed and sprout pixels", fontsize=22, fontweight='normal')

        self.size = 12
        font = {'size': self.size}
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.rc('font', **font)

        #
        # self.fig.subplots_adjust(top=0.90)
        # self.ax.set_title(self.lowerTitle)

    def getRGB(self, img):
        # Array slicing. OpenCV use BRG, but matplotlib use RGB
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        return r,g,b

    def plot3Dpoints(self, img1, img2, img3, img4):

        # Defining colors, from http://www.w3schools.com/tags/ref_colorpicker.asp
        backgroundColor = '#000000'
        seedColor = '#A37547'
        sproutColor = '#4A4A25'

        # However using colors that mach the bagground is not wise. Use contrast colors like red, green and blue

        # for c, m, image, label in [('r', 'o', img1, "Background"), ('g', 'o', img2, "Seed"), ('b', 'o', img3, "Sprout"), ('c', 'o', img4, "No labelled data")]:
        #     r, g, b = self.getRGB(image)
        #     self.ax.scatter(r, g, b, c=c, marker=m)
        #
        # self.ax.set_xlabel('R')
        # self.ax.set_ylabel('G')
        # self.ax.set_zlabel('B')
        # self.ax.set_xlim(0, 255)
        # self.ax.set_ylim(0, 255)
        # self.ax.set_zlim(0, 255)
        #
        # # # A hack for getting the legend.
        # red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
        # green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
        # blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
        # cyan_proxy = plt.Rectangle((0, 0), 1, 1, fc="c")
        # self.ax.legend([red_proxy,green_proxy, blue_proxy, cyan_proxy],["Background (" +str(img1.size/3) + ")",
        #                                                                 "Seed (" + str(img2.size/3) + ")",
        #                                                                 "Sprout (" + str(img3.size/3) + ")",
        #                                                                 "No labelled (" + str(img4.size/3) + ")"],
        #                                                                 bbox_to_anchor=(1.1, 1.05))


        for c, m, image, label in [('r', 'o', img1, "Background"), ('g', 'o', img2, "Seed"), ('b', 'o', img3, "Sprout")]:
            r, g, b = self.getRGB(image)
            self.ax.scatter(r, g, b, c=c, marker=m)

        self.ax.set_xlabel('R')
        self.ax.set_ylabel('G')
        self.ax.set_zlabel('B')
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
        self.ax.set_zlim(0, 255)

        # # A hack for getting the legend.
        red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
        green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
        blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
        self.ax.legend([red_proxy,green_proxy, blue_proxy],["Background (" +str(img1.size/3) + ")",
                                                            "Seed (" + str(img2.size/3) + ")",
                                                            "Sprout (" + str(img3.size/3) + ")"],
                                                            bbox_to_anchor=(1.1, 1.05))

        plt.show()

    def randrange(self, n, vmin, vmax):
        return (vmax-vmin)*np.random.rand(n) + vmin
