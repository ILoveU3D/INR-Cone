import numpy as np
import torch

dim_x, dim_y, dim_z = 512, 512, 16
input = torch.zeros(dim_x, dim_y, dim_z, 3)
value_xy = np.linspace(-1,1,dim_x)
value_z = np.linspace(-1,1,dim_z)
for x in range(dim_x):
    for y in range(dim_y):
        for z in range(dim_z):
            input[x,y,z,0] = value_xy[x]
            input[x,y,z,1] = value_xy[y]
            input[x,y,z,2] = value_z[z]
input.numpy().tofile("/media/wyk/wyk/Data/raws/input.raw")