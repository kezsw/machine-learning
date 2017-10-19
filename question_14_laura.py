import GPy
import math
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random

pi = math.pi

sample_size = 7
x = np.linspace(-pi,pi,sample_size)
x = x.reshape(-1,1)
mu_err = 0
var_err = 0.05
sigma_err = math.sqrt(0.05)
err = norm.pdf(x, loc=mu_err, scale=sigma_err)
Y = np.sin(x) + err

l = 6
mu = np.zeros(x.shape)
K = np.exp(-cdist(x,x)**2/l)

f = np.random.multivariate_normal(mu.flatten(),K,sample_size)

#plt.plot(x,f.T)
#plt.show()

x_star = np.linspace(0,0,sample_size)
x_star = x_star.reshape(-1,1)
for i in range(0, sample_size):
  x_star[i] = np.random.uniform(-pi,pi),



K1 = np.exp(-cdist(x_star,x)**2/l)
K2 = np.exp(-cdist(x_star,x_star)**2/l)
K3 = np.exp(-cdist(x,x_star)**2/l)

#print K1 == np.transpose(K3)


K1T = np.transpose(K1)
#print K1T.shape, K1T.shape, K2.shape, K3.shape, K.shape
Kinv = np.linalg.inv(K)
#print inverseK == K**-1
mid_cal = np.dot(K1T,Kinv)
print mid_cal.shape
mu_star = np.dot(mid_cal,Y)
cov_star = K2 - np.dot(mid_cal,K3)
print Y.shape, mu_star.shape, cov_star.shape
f_star = np.random.multivariate_normal(mu_star.flatten(),cov_star,sample_size)

#plt.plot(x_star,f_star.T)
plt.plot(x,f.T)
#plt.plot(x_star,np.sin(x)+f_star.T)
#plt.plot(x_star,f.T)
plt.show()
