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
import ConeBeamLayers.BeijingGeometry as geometry

device1 = "cuda:2"
device2 = "cuda:3"
for m in range(beijingVolumeSize[2]//2):
    model[m] = model[m].to(device1)
    model[m].train()
for m in range(beijingVolumeSize[2]//2,beijingVolumeSize[2]):
    model[m] = model[m].to(device2)
    model[m].train()
params = []
for m in range(360):
    params.append(geometry.parameters[m*3*21:(m*3+1)*21,:])
geometry.parameters = torch.stack(params).view(-1,12)

if __name__ == '__main__':
    projection = np.fromfile("/home/nv/wyk/Data/lz/projection2.raw", dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,144,80)
    projection[torch.isnan(projection)] = 0
    projection_ = []
    for m in range(360):
        projection_.append(projection[...,m*3*21:(m*3+1)*21,:,:])
    projection = torch.stack(projection_).reshape(1,1,360*21,144,80).to(device1)
    projection = projection[...,3:77]
    input = np.fromfile("/home/nv/wyk/Data/input.raw", dtype="float32")
    input = torch.from_numpy(input)
    bytelen = beijingVolumeSize[0]*beijingVolumeSize[1]
    input = input.reshape(bytelen,2)
    lossFunction = nn.MSELoss()
    optimizer = [torch.optim.Adam(m.parameters(), lr=2e-4) for m in model]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=50, gamma=0.8) for o in optimizer]
    output = torch.zeros(beijingVolumeSize[2], bytelen, 1).to(device1)
    tic = time.time()
    print("start")
    for e in range(31):
        for s in range(beijingVolumeSize[2]//2):
            output[s,...] = model[s](input.to(device1)).float()
        for s in range(beijingVolumeSize[2]//2,beijingVolumeSize[2]):
            output[s,...] = model[s](input.to(device2)).float().to(device1)
        output_projection = ForwardProjection.apply(output.reshape(1, 1,beijingVolumeSize[2],beijingVolumeSize[1],beijingVolumeSize[0]))
        loss = lossFunction(output_projection, projection)
        loss.backward()
        for s in range(beijingVolumeSize[2]):
            optimizer[s].step()
            optimizer[s].zero_grad()
            scheduler[s].step()
        if e%10==0:
            output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/lz/output.raw")
        print(f"loss:{loss.item()}")
        output.detach_()
        torch.cuda.empty_cache()
    with torch.no_grad():
        for s in range(beijingVolumeSize[2]//2):
            output[s,...] = model[s](input.to(device1)).float()
        for s in range(beijingVolumeSize[2]//2,beijingVolumeSize[2]):
            output[s,...] = model[s](input.to(device2)).float().to(device1)
        output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/lz/output.raw")
    print(f"time cost:{time.time() - tic}")
