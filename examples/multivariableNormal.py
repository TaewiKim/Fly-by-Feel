import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Generate grid of points
x, y, z = np.mgrid[-2000:2000:100, 0:3000:100, -5000:5000:100]

# Reshape the grid to be compatible with rv.pdf()
pos = np.empty(x.shape + (3,))
pos[...,0] = x
pos[...,1] = y
pos[...,2] = z

# Define the multivariate normal distribution
mu = [0, 1500, 5000]
cov = [[500000, 0, 0], [0, 500000, 0], [0, 0, 100000000]]
rv = multivariate_normal(mu, cov)

# Calculate the probability density function
k = rv.pdf(pos)

# Create a contour plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.contourf(y[0,:,:], z[0,:,:], k[0,:,:])  # Using the first z-slice for the contour plot
fig2.show()
