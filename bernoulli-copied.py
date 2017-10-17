import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


def posterior(a, b, X):
    a_n = a + X.sum()
    b_n = b + (X.shape[0] - X.sum())
    return beta.pdf(mu_test, a_n, b_n)


# parameters to generate data
mu = 0.2
N = 200

# generate some data
X = np.random.binomial(1, mu, N)
mu_test = np.linspace(0, 1, 100)

# now lets define our prior
a = 5.0
b = 5.0

# p(mu) = Beta(alpha,beta)
prior_mu = beta.pdf(mu_test, a, b)

# we have derived the posterior
a_n = a + X.sum()
b_n = b + (N - X.sum())
posterior_mu = beta.pdf(mu_test, a_n, b_n)

# create figure
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

# plot prior
ax.plot(mu_test,prior_mu,'g')
ax.fill_between(mu_test,prior_mu,color='green',alpha=0.3)
ax.set_xlabel('$\mu$')
ax.set_ylabel('$p(\mu|\mathbf{x})$')

# plt.plot(mu_test,posterior_mu,'r')

# # lets pick a random (uniform) point from the data # and update our assumption with this
# index = np.random.permutation(X.shape[0])
# for i in range(0,X.shape[0]):
#     y = posterior(a,b,X[:index[i]])
#     plt.plot(mu_test,y,'r',alpha=0.3)
plt.show()