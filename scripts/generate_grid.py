import numpy as np

size = (512,512)
grid = np.zeros(size)
grid[200:312:20,:] = 1
grid[:,200:312:20] = 1
grid[:100,:] = 0
grid[412:,:] = 0
grid[:,:100] = 0
grid[:,412:] = 0
grid.astype("float32").tofile("/home/nv/wyk/Data/grid.raw")
