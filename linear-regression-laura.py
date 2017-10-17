############################################
# Linear Regression
#
# Author: Aidan Scannell
# Date: October 2017
############################################

import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from numpy.linalg import inv

# parameters to generate data
mu = 0
sigma = 0.3
W = np.array([-1.3,0.5]) # [W1 W0]
W = W.reshape(-1,2)
# cov=t*np.eye(W.shape[1])

# covariance
t = 0.1

# generate data
x = np.ones([2,201])
x[1,:] = np.linspace(-1,1,201)
# x = np.linspace(-1,1,201)
# x = x.reshape(-1,1)

# calculate prior
prior_w = multivariate_normal(mean=[W.item(0),W.item(1)],cov=[[t,0],[0,t]]).pdf(x.T)
prior_w = prior_w.reshape(-1,1)

error = norm(mu,sigma).pdf(x)
Y = W.T * x + error

# create figure
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_xlabel('$w$')
ax.set_ylabel('$p(w|x,y)$')

# plot prior
ax.plot(x.T,prior_w,'b')

# posterior
# def posterior(Y):
#     m_n = 1/sigma * inv(1/sigma * x.T * x + (t*np.identity(x.size))) * x.T * Y
#     s_n_inv = 1/sigma * x.T * x + t
#     print j
#     j = j + 1
#     return norm(m_n,s_n_inv).pdf(x)

# m_n = 1/sigma * (inv(1/sigma * x.T.dot(x) + t*np.identity(x.shape[1]))).dot(x.T).dot(Y[:1])
# s_n_inv = 1/sigma * (x.T).dot(x) + t*np.identity(x.shape[1])
# posterior_w = norm(m_n,s_n_inv).pdf(x)
#
# ax.plot(x,posterior_w[:,100],'r')
# ax.plot(x,posterior_w[:,101],'g')
# ax.plot(x,posterior_w[:,102],'b')
# ax.plot(x,posterior_w[:,103],'c')
# plt.show()

# # randomly select points from data
# index = np.random.permutation(W.shape)
#
# # update assumption (set posterior (n) as prior (n+1)
# for i in range(0,Y.shape[0]):
#     # plot posterior
#     y = posterior(Y[:index[i]])
#     ax.plot(x,y,'r', alpha=0.3)