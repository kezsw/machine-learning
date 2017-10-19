############################################
# Linear Regression
#
# Author: Aidan Scannell
# Date: October 2017
############################################

import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

# parameters to generate data
mu = 0
sigma = 0.3
W = np.array([-1.3,0.5]) # [W1 W0]
W = W.reshape(-1,1)
t = 0.8 # covariance

# generate data
x = np.linspace(-1,1,201)
x = x.reshape(-1,1)

# calculate prior
prior_w = multivariate_normal(mean=W.flatten(),cov=[[t,0],[0,t]])
# prior_w = prior_w.reshape(-1,1)

# plt.plot(x,prior_w)
# plt.show()

# contour plot
w0, w1 = np.mgrid[-2:2:.01, -2:2:.01]
pos = np.empty(w0.shape + (2,))
pos[:, :, 0] = w0; pos[:, :, 1] = w1
plt.contourf(w0, w1, prior_w.pdf(pos))
plt.show()

# wx = w0[:,1]
# wx = wx.reshape(1,-1)
# wy = w1[1,:]
# wy = wy.reshape(1,-1)
# Axes3D.plot_surface(wx,wy,prior_w.pdf(pos))

# create Y by adding column of 1's to x
error = np.random.normal(mu,sigma,x.shape)
# Y = (W.T).dot(x) + error
w0 = 0.5
w1 = -1.3
Y = x*w0 + w1
Y = Y + error

# Display original data
plt.scatter(x,Y)
plt.show()

# posterior
# m_n = 1/(sigma**2) * (inv(1/(sigma**2) * x.T.dot(x) + t*np.identity(x.shape[1]))).dot(x.T).dot(Y)
def posterior(x,Y):
    xtx = x.T.dot(x)
    t2i = t * np.identity(x.shape[1])
    sumMn = (1 / (sigma ** 2) * xtx) + t2i
    invMn = inv(sumMn)
    xty = (x.T).dot(Y)
    m_n = 1 / (sigma ** 2) * invMn.dot(xty)
    s_n_inv = 1/(sigma**2) * (x.T).dot(x) + t*np.identity(x.shape[1])
    posterior_w = multivariate_normal(mean=[m_n.item(0)],cov=[s_n_inv.item(0)])
    return posterior_w.pdf(x)

# posterior_w = posterior_w.reshape(-1,1)

# contour plot
# w0, w1 = np.mgrid[-2:2:.01, -2:2:.01]
# pos = np.empty(w0.shape + (2,))
# pos[:, :, 0] = w0; pos[:, :, 1] = w1
# plt.contourf(w0, w1, posterior_w)
# plt.show()

x = x[:100,:]
Y = Y[:100,:]
plt.plot(x,posterior(x,Y),'r',alpha=0.3)
plt.show()