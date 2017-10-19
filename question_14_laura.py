import GPy
import math
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

pi = math.pi

sample_size = 7
x = np.linspace(-pi,pi,sample_size)
x = x.reshape(-1,1)
mu_err = 0
var_err = 0.05
sigma_err = math.sqrt(0.05)
err = norm.pdf(x, loc=mu_err, scale=sigma_err)
Y = np.sin(x) + err

l = 80
mu = np.zeros(x.shape)
K = np.exp(-cdist(x,x)**2/l)

f = np.random.multivariate_normal(mu.flatten(),K,sample_size)


x_star = np.linspace(-1,1,sample_size)
x_star = x_star.reshape(-1,1)
K1 = np.exp(-cdist(x_star,x)**2/l)
K2 = np.exp(-cdist(x_star,x_star)**2/l)
K3 = np.exp(-cdist(x,x_star)**2/l)

K1T = np.transpose(K1)
inverseK = K**-1#np.linalg.inv(K)
test = K1.dot(inverseK).dot(Y)
test2= K2 - K1.dot(inverseK).dot(K3)
print test.shape, test2.shape


print x_star.shape, K1.shape, K1T.shape, K.shape

print inverseK == K**-1
inverseKY = inverseK.dot(Y)
inverseKK3 = inverseK.dot(K3)
mu_star = K1T.dot(inverseKY)
cov_star= K2 - K1T.dot(inverseKK3)
print cov_star
p = np.random.multivariate_normal(test.flatten(),test2,sample_size)

#plt.plot(x,f.T)
#plt.show()
