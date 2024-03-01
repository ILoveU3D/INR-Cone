# The final INR Recon for Nanovision 24
# For the CUDA extensions get compiled，do not run this file directly. Run `sh run.sh nerf_cone_real.py` instead. 
# By Wangyukang Feb 28, 2024

import numpy as np
import torch
import torch.nn as nn
import time
import tqdm
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

def build_coordinate_test(L):
    input = torch.zeros(L, L, 2)
    value_xy = np.linspace(-1,1,L)
    for x in range(L):
        for y in range(L):
            input[x,y,0] = value_xy[x]
            input[x,y,1] = value_xy[y]
    return input.reshape(-1, 2).to(device)

if __name__ == '__main__':
    projectionPath = "/home/nv/wyk/Data/lz/projection.raw"
    outputPath = "/home/nv/wyk/Data/output.raw"
    projection = np.fromfile(projectionPath, dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,128,80).to(device)
    projection[torch.isnan(projection)] = 0
    projection = projection[...,2:78]
    bytelen = beijingVolumeSize[0]*beijingVolumeSize[1]
    input = build_coordinate_test(beijingVolumeSize[0])
    lossFunction = nn.MSELoss()
    optimizer = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in model]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=50, gamma=0.95) for o in optimizer]
    tic = time.time()
    with tqdm.trange(51) as epochs:
        for e in epochs:
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
                output.detach().cpu().numpy().astype("float32").tofile(outputPath)
            epochs.set_description(f"Learning projections ...")
            epochs.set_postfix({"loss": loss.item()})
    print("Training successfully. Now we are going to render...")
    output = torch.zeros(beijingVolumeSize[2], bytelen, 1).to(device)
    for s in range(beijingVolumeSize[2]):
        output[s,...] = model[s](input).float()
    output.detach().cpu().numpy().astype("float32").tofile(outputPath)
    print(f"Success. Render output is saved in {outputPath}. Total time cost:{time.time() - tic}s")