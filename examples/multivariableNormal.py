import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[-360:360:10, -360:360:10]
pos = np.dstack((x, y))
mu = [0, 0]
cov = [[3000, 0], [0, 3000]]
rv = multivariate_normal(mu, cov)
z = rv.pdf(pos)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, z)
plt.show()