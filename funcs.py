import math
from time import time

import numpy as np

from scipy.spatial import KDTree
from scipy.sparse.linalg import LinearOperator, cg

from examples.example_1 import target, faults

def distance(p1, p2):
  return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def basis(r, a = 1.):
  return (4 * r / a + 1) * (1 - r / a) ** 4

def line(p1, p2):
  if abs(p2[0] - p1[0]) >= 1e-4:
    k = (p2[1] - p1[1])/(p2[0] - p1[0])
    b = (p1[1]*p2[0] - p2[1]*p1[0])/(p2[0] - p1[0])

    return (k, b)

  return None

def lines_intersection(k1, b1, k2, b2):
  if abs(k2 - k1) >= 1e-4:
    x = (b1 - b2)/(k2 - k1)
    y = (k2*b1 - k1*b2)/(k2 - k1)

    return (x, y)

  return None

def point_in_range(p, p1, p2):
  t = (p[0] - p1[0])/(p2[0] - p1[0])

  if t >= 0 and t <= 1:
    return True

  return False

def calculate_point(p1, p2, r, l):
  return (
    p1[0] + r * (p2[0] - p1[0]) / l,
    p1[1] + r * (p2[1] - p1[1]) / l
  )

def exclude_points(pr, p1, p2, tree, rf, points, res):
  line_data = line(p1, p2)

  found = tree.query_ball_point(pr, rf)

  up = []
  down = []

  for i in found:
    p = points[i]

    if line_data == None:
      if p[0] >= p1[0]:
        up.append(i)
      else:
        down.append(i)
    else:
      k, b = line_data

      y = k * p[0] + b

      if p[1] >= y:
        up.append(i)
      else:
        down.append(i)

  for i in up:
    pp1 = points[i]

    for j in down:
      pp2 = points[j]

      line_data_2 = line(pp1, pp2)      

      exclude = False  

      if line_data == None:
        if line_data_2 == None:
          if pp1[0] == p1[0]:
            y1 = p1[1]
            y2 = p2[1]

            if p1[1] > p2[1]:
              y1 = p2[1]
              y2 = p1[1]

            if (pp1[1] >= y1 and pp1[1] <= y2) or (pp2[1] >= y1 and pp2[1] <= y2):
              exclude = True
        else:
          k2, b2 = line_data_2

          y = k2 * p1[0] + b2

          y1 = p1[1]
          y2 = p2[1]

          if p1[1] > p2[1]:
            y1 = p2[1]
            y2 = p1[1]

          if y >= y1 and y <= y2:
            exclude = True
      else:
        k, b = line_data

        if line_data_2 == None:
          y = k * pp1[0] + b
          exclude = point_in_range((pp1[0], y), p1, p2)
        else:
          k2, b2 = line_data_2

          p = lines_intersection(k, b, k2, b2)

          if p == None:
            exclude = point_in_range(pp1, p1, p2) or point_in_range(pp2, p1, p2)
          else:
            exclude = point_in_range(p, p1, p2)

      if exclude:
        res[i].discard(j)
        res[j].discard(i)  

def faults_exclude_points(faults, tree, rf, points, res, ri):
  for p1, p2 in faults:
    l = distance(p1, p2)
      
    exclude_points(p1, p1, p2, tree, rf, points, res)

    r = ri

    while r < l:
      pr = calculate_point(p1, p2, r, l)

      exclude_points(pr, p1, p2, tree, rf, points, res)

      r += ri

    exclude_points(p2, p1, p2, tree, rf, points, res)  

def rbf(points):
  m = len(points)

  A = np.zeros((m, m))
  f = np.zeros(m)

  for i in range(0, m):
    f[i] = target(points[i][0], points[i][1])

    for j in range(0, m):
      p1 = points[i]
      p2 = points[j]

      r = distance(
        [p1[0], p1[1], 0], 
        [p2[0], p2[1], 0]
      )

      A[i][j] = basis(r)

  return np.linalg.solve(A, f)

def rbf_interpolant(b, points, x, y):
  m = len(points)
  n = x.shape[0]

  z = np.zeros(x.shape)
  
  for k in range(0, m):
    for i in range(0, n):
      for j in range(0, n):
        r = distance(
          (x[i, j], y[i, j], 0), 
          [points[k][0], points[k][1], 0]
        )

        z[i, j] += b[k] * basis(r)

  return z

def cs_rbf(points, faults, ri):
  n = len(points)

  tree = KDTree(points, copy_data=True)

  max_count = 0
  min_count = n

  new_points = []

  rf = 2 * ri / math.sqrt(3)

  for p in points:    
    count = tree.query_ball_point(p, ri, return_length=True)

    if count >= 2:
      max_count = max(max_count, count)
      min_count = min(min_count, count)

      new_points.append(p)

  points = new_points
  
  n = len(points)

  print('Before fault detection:')
  print(f'points = {n}, max count = {max_count}, min count = {min_count}, ri = {ri}')

  tree = KDTree(points, copy_data=True)

  res = []

  for p in points:
    found = tree.query_ball_point(p, ri)

    res.append(set(found))

  faults_exclude_points(faults, tree, rf, points, res, ri)

  max_count = 0
  min_count = n
  new_points = []

  for i in range(0, n):
    s = res[i]

    if len(s) > 1:
      min_count = min(min_count, len(s))
      max_count = max(max_count, len(s))
      new_points.append(points[i])

  points = new_points

  n = len(points)

  print('After fault detection:')
  print(f'points = {n}, max count = {max_count}, min count = {min_count}, ri = {ri}')

  tree = KDTree(points, copy_data=True)

  res = []
  fb = []

  for p in points:
    fb.append(target(p[0], p[1]))

    found = tree.query_ball_point(p, ri)

    res.append(set(found))

  faults_exclude_points(faults, tree, rf, points, res, ri)
  
  def mv(v):
    f = np.zeros(n)

    for i in range(n):
      for j in res[i]:
        a = basis(
          distance(points[i], points[j]),
          ri
        )

        f[i] += a * v[j]

    return f

  A = LinearOperator((n, n), matvec=mv)

  start = time()

  b = cg(A, fb)

  finish  =  time()

  print(f'time = {finish - start}, iterations={b[1]}')

  return points, tree, b

def cs_rbf_interpolant(tree: KDTree, b, points, x, y, ri):
  n, m = x.shape

  z = np.zeros(x.shape)

  for i in range(0, n):
    for j in range(0, m):
      found = tree.query_ball_point([x[i, j], y[i, j]], ri)    

      for k in found:        
        p = points[k]
        z[i, j] += b[k] * basis(distance((x[i, j], y[i, j]), p), ri)

  return z  
