import math

def target(x, y):  
  if x >= -0.5 and x <= 0.5 and y >= -0.5 and y <= 0.5:
    if y >= x:
      return 2
    else:
      return 1

  return 0

faults = [(
  (-0.5, -0.5),
  (-0.5, 0.5)
), (
  (-0.5, 0.5),
  (0.5, 0.5)
), (
  (-0.5, -0.5),
  (0.5, 0.5)
), (
  (-0.5, -0.5),
  (0.5, -0.5)
), (
  (0.5, -0.5),
  (0.5, 0.5)
)]
