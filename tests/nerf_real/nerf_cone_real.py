import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
import os
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from ConeBeamLayers.BeijingGeometry import ForwardProjection
from model import model
from options import beijingVolumeSize

device = "cuda:2"
for m in model:
    m = m.to(device)
    m.train()

if __name__ == '__main__':
    # label = np.fromfile("/data/mcl/data/AAPM/1.raw", dtype="float32")
    # label = np.reshape(label, [1, 1, 200, 512, 512])
    # label = label[...,0:63,:,:]
    # label /= np.max(label)
    # label = torch.from_numpy(label).to(device)
    # projection = ForwardProjection.apply(label)
    projection = np.fromfile("/home/nv/wyk/Data/lz/projection2.raw", dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,144,80).to(device)
    projection[torch.isnan(projection)] = 0
    # projection = projection[...,1:79]
    input = np.fromfile("/home/nv/wyk/Data/input.raw", dtype="float32")
    input = torch.from_numpy(input)
    bytelen = beijingVolumeSize[0]*beijingVolumeSize[1]
    input = input.reshape(bytelen,2).to(device)
    lossFunction = nn.MSELoss()
    optimizer = [torch.optim.Adam(m.parameters(), lr=3e-4) for m in model]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=50, gamma=0.8) for o in optimizer]
    tic = time.time()
    print("start")
    for e in range(31):
        output = torch.zeros(beijingVolumeSize[2], bytelen, 1).to(device)
        for s in range(beijingVolumeSize[2]):
            output[s,...] = model[s](input).float()
        output_projection = ForwardProjection.apply(output.reshape(1, 1,beijingVolumeSize[2],beijingVolumeSize[1],beijingVolumeSize[0]))
        loss = lossFunction(output_projection, projection)
        loss.backward()
        for s in range(beijingVolumeSize[2]):
            optimizer[s].step()
            optimizer[s].zero_grad()
            scheduler[s].step()
        if e%1==0:
            output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/lz/output.raw")
        print(f"loss:{loss.item()}")
    output = torch.zeros(beijingVolumeSize[2], bytelen, 1).to(device)
    for s in range(beijingVolumeSize[2]):
        output[s,...] = model[s](input).float()
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/lz/output.raw")
    print(f"time cost:{time.time() - tic}")
