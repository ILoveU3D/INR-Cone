import numpy as np
import torch

dim_x, dim_y, dim_z = 512, 512, 16
input = torch.zeros(dim_x, dim_y, 2)
value_xy = np.linspace(-1,1,dim_x)
value_z = np.linspace(-1,1,dim_z)
for x in range(dim_x):
    for y in range(dim_y):
        input[x,y,0] = value_xy[x]
        input[x,y,1] = value_xy[y]
input.numpy().tofile("/home/nv/wyk/Data/input.raw")
