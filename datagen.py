#%%
import numpy as np
import math

def sin(seed = None, size = 1000, granularity=30, cont=0.01, abrupt=100):
  if seed:
    np.random.seed(seed=seed)
  start = np.random.rand()
  data = []
  for x in range(size):
    if np.random.rand() < 1/abrupt:
      start = start + np.random.rand()
    data.append([x, math.sin(start + x/(2*math.pi*granularity)) + np.random.normal(scale=0.05) + cont])
  return data

def stairs(seed = None, size = 1000, granularity=30, cont=0.01, abrupt=100):
  if seed:
    np.random.seed(seed=seed)
  start = np.random.rand()*100
  data = []
  for x in range(size):
    if np.random.rand() < 1/abrupt:
      start = start + (200 + np.random.rand() * 200)/granularity
    data.append([x, start + x/granularity + np.random.normal(scale=0.05) + cont])
  return data


#%%
print(stairs())


#%%
