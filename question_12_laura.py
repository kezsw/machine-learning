import numpy as np
import pylab
import math
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

x = np.linspace(-1,1,201)
x = x.reshape(-1,1)

x = np.ones([2,201])
x[1,:] = np.linspace(-1,1,201)

W = np.array([-1.3,0.5]) # [W1 W0]
W = W.reshape(-1,1)

WT = np.transpose(W)
x_shape = x.shape
W_shape = W.shape
WT_shape = WT.shape

#if WT_shape[1] == x_shape[0]:
#	mu = WT.dot(x)
#	mu_shape = mu.shape
#	print mu_shape, mu
#else:
#	print "Error"

mu_err = 0
var_err = 0.3
sigma_err = math.sqrt(var_err)
err = norm(mu_err,sigma_err).pdf(x)
err_shape = err.shape
Y = (W.T).dot(x) + err
Y_shape = Y.shape


t = 0.5
#cov = np.eye(x_shape[1])*t
#cov_shape = cov.shape
#print cov_shape
#prior_w = multivariate_normal(mean=mu.flatten(),cov=cov)

w0, w1 = np.mgrid[-2:2:.01, -2:2:.01]
pos = np.empty(w0.shape + (2,))
pos[:, :, 0] = w0; pos[:, :, 1] = w1
prior_w = multivariate_normal(W.flatten(), [[t, 0], [0, t]])
plt.contourf(w0, w1, prior_w.pdf(pos))
plt.show()
