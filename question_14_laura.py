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


l = 0.1
mu = np.zeros(x.shape)
K = np.exp(-cdist(x,x)**2/l)

f = np.random.multivariate_normal(mu.flatten(),K,sample_size)
Y = np.sin(x) + err


#plt.plot(x,f.T)
#plt.show()

x_star = np.linspace(0,0,sample_size)
#x_star = [-1.82447444; -0.51488409; 1.50483088; 1.34880768;-2.93171744; -2.29747605; -0.70214939]
x_star = x_star.reshape(-1,1)
#for i in range(0, sample_size):
#  x_star[i] = np.random.uniform(-pi,pi),
#print x_star
#x_star = x
K1 = np.exp(-cdist(x_star,x)**2/l)
K2 = np.exp(-cdist(x_star,x_star)**2/l)
K3 = np.exp(-cdist(x,x_star)**2/l)

#print K1 == np.transpose(K3)


K1T = np.transpose(K1)
#print K1T.shape, K1T.shape, K2.shape, K3.shape, K.shape
Kinv = np.linalg.inv(K)
#print inverseK == K**-1
mu_star = np.dot(np.dot(K1T,Kinv),f)
cov_star = np.subtract(K2, np.dot(np.dot(K1T,Kinv),K3))
#f_star = np.random.multivariate_normal(mu_star.flatten(),cov_star,sample_size)

print cov_star

# xlist = np.linspace(0, 250, sample_size) 
# ylist = np.linspace(250, 0, sample_size)
# X, Y = np.meshgrid(xlist, ylist)
# Z = cov_star
# plt.figure()
# cp = plt.contourf(X, Y, cov_star)
# plt.colorbar(cp)
# plt.title('Filled Contours Plot')
# plt.xlabel('x (cm)')
# plt.ylabel('y (cm)')
# plt.show()


xlist = np.linspace(0, 250, sample_size) 
ylist = np.linspace(4, 0, sample_size)
X, Y = np.meshgrid(xlist, ylist)
Z = mu
plt.figure()
cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()
