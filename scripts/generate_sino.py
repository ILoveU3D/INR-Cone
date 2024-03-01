import numpy as np
import torch
import os
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from ConeBeamLayers.BeijingGeometry import ForwardProjection
import ConeBeamLayers.BeijingGeometry as geometry

if __name__ == '__main__':
    label = np.fromfile("/home/nv/wyk/Data/balls.raw", dtype="float32")
    label = np.reshape(label, [1, 1, 16, 512, 512])
    # label = label[:,:, 24:40, ...]
    label = torch.from_numpy(label).cuda()
    # label /= np.max(label)
    geometry.parameters[:,0:3] += np.random.rand(geometry.parameters.shape[0], 3)
    projection = ForwardProjection.apply(label)
    projection.cpu().numpy().tofile("/home/nv/wyk/Data/ball_sino.raw")
