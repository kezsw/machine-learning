import numpy as np
import pylab
import math
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

pi = math.pi

x = np.linspace(-5,5,200)#np.linspace(-pi,pi,7)
x = x.reshape(-1,1)

mu = np.zeros(x.shape)
K = np.exp(-cdist(x,x)**2/5)
f = np.random.multivariate_normal(mu.flatten(),K,10)


x2 = np.linspace(-pi,pi,7)
x2= x.reshape(-1,1)

mu2 = np.zeros(x2.shape)
K2 = np.exp(-cdist(x2,x2)**2/5)
mu1 = 0
var3 = 0.5
sigma3 = math.sqrt(var3)
error norm.pdf(X, loc=mu, scale=sigma1)
y = np.sin(x2) + error
y= y.reshape(-1,1)

#y2 = np.sin(X2)

plt.plot(x2,y)
plt.show()