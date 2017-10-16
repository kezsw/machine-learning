import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# parameters to generate data
mu = 0
sigma = 0.3
N = 200

cov = np.array([[0.3,0],[0,0.3]])

# generate data
X = np.linspace(-1,1,201)
x = np.arange(-1.00, 1.00, 0.01);
W = np.array([-1.3,0.5])
W =  np.transpose(W)

error = np.random.normal(0,0.3,len(x))
# create prior
prior_W_1 = np.random.normal(W[1],0.9,len(x))
prior_W_2 = np.random.normal(W[1],0.5,len(x))
prior_W_3 = np.random.normal(W[1],0.1,len(x))

y = W[0]*x + W[1] + error;
#prior_W = np.random.multivariate_normal(W, cov, X.shape[0])

# visualise prior
line1, = plt.plot(x,error, label="error")
line2, = plt.plot(x,prior_W_1, label="cov = 0.9")
line3, = plt.plot(x,prior_W_2, label="cov = 0.5")
line4, = plt.plot(x,prior_W_3, label="cov = 0.1")
line5, = plt.plot(x,y, label="y")
plt.legend(handles=[line1, line2,line3,line4,line5])
plt.show()

# # posterior
# def posterior:
#     m_n = s_n * ( s_0_inv * m_0 + beta * phi_T * t)
#     s_n_inv =