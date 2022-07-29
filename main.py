import math
from time import time

import numpy as np

import matplotlib.pyplot as plt

import funcs

from examples.example_1 import faults

x = np.outer(np.linspace(-1, 1, 101), np.ones(101))
y = np.outer(np.linspace(-1, 1, 101), np.ones(101)).T

n, m = x.shape

z = np.zeros(x.shape)

for i in range(0, n):
  for j in range(0, m):
    z[i, j] = funcs.target(x[i, j], y[i, j])

max_az = np.max(np.absolute(z))

fig = plt.figure()

n = 1024

ri = 0.8 # 8 * math.sqrt(n_min / n)

points = np.random.normal(0, 1, (n, 2))

plt.title(f'RBF VS CS-RBF, n={n}')

ax = fig.add_subplot(1, 4, 1)
ax.set_xticks(np.arange(-1, 1.1, 0.4))
ax.set_yticks(np.arange(-1, 1.1, 0.2))
ax.scatter(points[:, 0], points[:, 1])
ax.grid()
ax.set_title('Scatter plot')

ax = fig.add_subplot(1, 4, 2, projection='3d')
ax.plot_surface(x, y, z,cmap='plasma')
ax.set_title('Surface plot')

""" start = time()
print('rbf start: ', start)

new_points, tree, b = funcs.cs_rbf(points, [], ri)
z_cs_rbf = funcs.cs_rbf_interpolant(tree, b[0], new_points, [], x, y, ri)
mre = np.max(np.absolute(z-z_cs_rbf)) / max_az

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

ax = fig.add_subplot(1, 4, 3, projection='3d')
ax.plot_surface(x, y, z_cs_rbf, cmap='plasma')
ax.set_title(f'Without Faults, mre={round(mre, 2)}, time={round(finish - start, 2)}')
 """
start = time()
print('cs rbf start: ', start)

new_points, tree, b = funcs.cs_rbf(points, faults, ri)
z_cs_rbf = funcs.cs_rbf_interpolant(tree, b[0], new_points, faults, x, y, ri)
mre = np.max(np.absolute(z-z_cs_rbf)) / max_az

finish = time()
print('cs rbf finish: ', finish, ', time: ', finish - start)

ax = fig.add_subplot(1, 4, 4, projection='3d')
ax.plot_surface(x, y, z_cs_rbf, cmap='plasma')
ax.set_title(f'With Faults, mre={round(mre, 2)}, time={round(finish - start, 2)}')

plt.show()
