import matplotlib.pyplot as plt
import time
import random
import numpy as np

ysample = np.random.random(100)

xdata = []
ydata = []

plt.show()

axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-1, 1)
line, = axes.plot(xdata, ydata, 'r-')

for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    #line.set_xdata(xdata)
    #line.set_ydata(ydata)
    plt.plot(xdata, ydata)
    #plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)

# add this if you don't want the window to disappear at the end
plt.show()