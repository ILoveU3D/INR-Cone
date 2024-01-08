import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
import time
import os
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from ConeBeamLayers.BeijingGeometry import ForwardProjection
from model import model
from options import beijingVolumeSize

device = "cuda:3"
for m in model:
    m = m.to(device)
    m.train()

if __name__ == '__main__':
    label = np.fromfile("/data/mcl/data/AAPM/1.raw", dtype="float32")
    label = np.reshape(label, [1, 1, 200, 512, 512])
    label = label[...,0:63,:,:]
    label /= np.max(label)
    label = torch.from_numpy(label).to(device)
    projection = ForwardProjection.apply(label)
    # projection = np.fromfile("/home/nv/wyk/Data/lz/projection.raw", dtype="float32")
    # projection = torch.from_numpy(projection).reshape(1,1,1080*21,128,80).to(device)
    input = np.fromfile("/home/nv/wyk/Data/input.raw", dtype="float32")
    input = torch.from_numpy(input)
    input = input.reshape(512*512,2).to(device)
    lossFunction = nn.MSELoss()
    optimizer = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in model]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=100, gamma=0.95) for o in optimizer]
    tic = time.time()
    for e in range(500):
        output = torch.zeros(beijingVolumeSize[2], 512*512, 1).to(device)
        for s in range(beijingVolumeSize[2]):
            output[s,...] = model[s](input).float()
        output_projection = ForwardProjection.apply(output.reshape(1, 1,beijingVolumeSize[2],512,512))
        loss = lossFunction(output_projection, projection)
        loss.backward()
        for s in range(beijingVolumeSize[2]):
            optimizer[s].step()
            optimizer[s].zero_grad()
            scheduler[s].step()
        if e%100==0:
            output[36,...].detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/lz/output.raw")
            print(f"loss:{loss.item()}")
    output = torch.zeros(beijingVolumeSize[2], 512*512, 1).to(device)
    for s in range(beijingVolumeSize[2]):
        output[s,...] = model[s](input).float()
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/lz/output.raw")
    print(f"time cost:{time.time() - tic}")
