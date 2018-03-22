import numpy as np
import matplotlib.pyplot as plt
from pylab import *

x = np.arange(1, 51)
a = np.load('lmserloss2_200000.npy')
b = np.load('lmserloss1_200000.npy')
c = np.load('biministloss_20000.npy')
print(a)
print(b)
print(c)
'''
d = np.load('autoloss_5000.npy')
plot(x, a, color="blue", linewidth=2.5, linestyle="-", label="lmser2")
plot(x, b, color="red",  linewidth=2.5, linestyle="-", label="lmser1")
plot(x, c, color="green",  linewidth=2.5, linestyle="-", label="biminist")
plot(x, d, color="yellow",  linewidth=2.5, linestyle="-", label="autoencoder")
plt.legend(loc='upper left')
plt.show()

'''
