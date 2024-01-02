import numpy as np
import torch

size = (512,512)
vectors = [torch.arange(0, s) for s in size]
grids = torch.meshgrid(vectors)
grid = torch.stack(grids)  # y, x, z
grid = torch.unsqueeze(grid, 0)  # add batch
grid = grid.type(torch.FloatTensor)
grid = grid.permute(0,2,3,1)[..., [1,0]]
print(grid.shape)
grid.numpy().astype("float32").tofile("/home/nv/wyk/Data/grid.raw")
