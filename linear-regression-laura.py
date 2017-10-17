import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from numpy.linalg import inv

# parameters to generate data
mu = 0
sigma = 0.3
W = np.array([-1.3,0.5]) # [W1 W0]
W = W.reshape(-1,2)
W = W.T
# cov=t*np.eye(W.shape[1])

# covariance
t = 0.8

# generate data
x = np.ones([2,201])
x[0,:] = np.linspace(-1,1,201)

prior_w = multivariate_normal(mean=[W.item(0),W.item(1)],cov=[[t,0],[0,t]]).pdf(x.T)
prior_w = prior_w.reshape(-1,1)

plt.plot(x.T,prior_w)
plt.show()

# error = np.random.normal(0,0.1,x.shape) # noise
error = norm(mu,sigma).pdf(x)

Y = np.ones(x.shape)*W[0]* x + W[1] + error

# posterior
# def posterior(Y):
#     m_n = 1/sigma * inv(1/sigma * x.T * x + (t*np.identity(x.size))) * x.T * Y
#     s_n_inv = 1/sigma * x.T * x + t
#     print j
#     j = j + 1
#     return norm(m_n,s_n_inv).pdf(x)

# create figure
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_xlabel('$w$')
ax.set_ylabel('$p(w|x,y)$')

# plot prior
ax.plot(x,prior_w,'b')


m_n = 1/sigma * inv(1/sigma * x.T * x + t*np.identity(x.shape[0])) * x.T * Y
s_n_inv = 1/sigma * x.T * x + t
posterior_w = norm(m_n,s_n_inv).pdf(x)

ax.plot(x,posterior_w[:,100],'r')
ax.plot(x,posterior_w[:,101],'g')
ax.plot(x,posterior_w[:,102],'b')
ax.plot(x,posterior_w[:,103],'c')
plt.show()

# # randomly select points from data
# index = np.random.permutation(W.shape)
#
# # update assumption (set posterior (n) as prior (n+1)
# for i in range(0,Y.shape[0]):
#     # plot posterior
#     y = posterior(Y[:index[i]])
#     ax.plot(x,y,'r', alpha=0.3)

# error = np.random.normal(0,0.000003,x.shape) # noise
#
# Y = np.ones(x.shape)*W[0]
# Y = Y * x
# Y = Y + W[1] + error
# x = np.arange(-1.00, 1.00, 0.01)
# W = np.array([-1.3,0.5])
# W =  np.transpose(W)
# w_test = np.linspace(0,1,100)
# w_test = w_test.reshape(-1,1)

# error = np.random.normal(0,0.3,len(x))
# create prior
# prior_W_1 = np.random.normal(W[1],0.9,len(x))
# prior_W_2 = np.random.normal(W[1],0.5,len(x))
# prior_W_3 = np.random.normal(W[1],0.1,len(x))

# y = W[0]*x + W[1] + error;
#prior_W = np.random.multivariate_normal(W, cov, X.shape[0])

# visualise prior
# line1, = plt.plot(x,error, label="error")
# line2, = plt.plot(x,prior_W_1, label="cov = 0.9")
# line3, = plt.plot(x,prior_W_2, label="cov = 0.5")
# line4, = plt.plot(x,prior_W_3, label="cov = 0.1")
# line5, = plt.plot(x,y, label="y")
# plt.legend(handles=[line1, line2,line3,line4,line5])
# plt.plot(w_test,prior_W_1)
# plt.show()

# # posterior
# def posterior:
#     m_n = 1/sigma * (1/sigma * x.T * x + cov) x.T * Y
#     s_n_inv = 1/sigma * x.T * x + t
#
# # create figure
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# ax.set_xlabel('$w$')
# ax.set_ylabel('$p(f|x,y)$')
#
# # plot prior
# ax.plot(mu_test,prior_mu,'b')
# ax.fill_between(mu_test,prior_mu,color='blue',alpha=0.3)
#
# # randomly select points from data
# index = np.random.permutation(X.shape[0])
#
# # update assumption (set posterior (n) as prior (n+1)
# for i in range(0,X.shape[0]):
#     # plot posterior
#     y = posterior(a,b,X[:index[i]])
#     ax.plot(mu_test,y,'r', alpha=0.3)
#
# plt.show()