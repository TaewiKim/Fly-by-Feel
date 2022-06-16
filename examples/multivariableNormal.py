import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[-1000:1000:100, -1000:1000:100]
pos = np.dstack((x, y))
rv = multivariate_normal([0, 0], [[100000, 0], [0, 500000]])
z = rv.pdf(pos)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, z)
plt.show()