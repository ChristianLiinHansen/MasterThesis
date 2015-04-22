from pylab import *
import time

ion()
fig = figure()
ax1 = fig.add_subplot(611)
ax2 = fig.add_subplot(612)
ax3 = fig.add_subplot(613)
ax4 = fig.add_subplot(614)
ax5 = fig.add_subplot(615)
ax6 = fig.add_subplot(616)

x = arange(0,2*pi,0.01)
y = sin(x)
line1, = ax1.plot(x, y, 'r-')
line2, = ax2.plot(x, y, 'g-')
line3, = ax3.plot(x, y, 'y-')
line4, = ax4.plot(x, y, 'm-')
line5, = ax5.plot(x, y, 'k-')
line6, = ax6.plot(x, y, 'p-')

# turn off interactive plotting - speeds things up by 1 Frame / second
plt.ioff()


tstart = time.time()               # for profiling
for i in arange(1, 200):
    line1.set_ydata(sin(x+i/10.0))  # update the data
    line2.set_ydata(sin(2*x+i/10.0))
    line3.set_ydata(sin(3*x+i/10.0))
    line4.set_ydata(sin(4*x+i/10.0))
    line5.set_ydata(sin(5*x+i/10.0))
    line6.set_ydata(sin(6*x+i/10.0))
    draw()                         # redraw the canvas

print 'FPS:' , 200/(time.time()-tstart)