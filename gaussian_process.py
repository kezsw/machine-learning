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
var = 0.5
sigma = math.sqrt(var)
#error = norm.pdf(X, loc=mu, scale=sigma)
f = np.random.multivariate_normal(mu.flatten(),K,10)

#y = np.sin(x) + error

#y2 = np.sin(X2)

plt.plot(x,f.T)
plt.show()