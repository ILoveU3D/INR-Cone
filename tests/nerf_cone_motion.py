###
# Nerf重建北京抖动数据
###

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import time
import os
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from ConeBeamLayers.BeijingGeometry import ForwardProjection
from model import model

device = "cuda:3"
for m in model:
    m = m.to(device)
    m.train()

def translate(target, rotation, t):
    shape = target.shape
    target = target.view(24, int(1080*128/24), 80*21)
    result = torch.zeros_like(target)
    rotation = rotation.view(2, 2, 24).permute(2, 0, 1).contiguous()
    for angle in range(24):
        h, w = target[angle, ...].shape
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        input_coords = torch.stack((grid_x, grid_y), dim=-1).float().view(-1,2).to(device)
        rot_matrix = rotation[angle, ...]
        modify = torch.matmul(input_coords, rot_matrix).squeeze(1) + t[angle, ...]
        result[angle, ...] = F.grid_sample(target[angle, ...].unsqueeze(0).unsqueeze(0), modify.reshape(1,h,w,2)).squeeze()
    return result.reshape(shape) + target.reshape(shape) * 0.1

if __name__ == '__main__':
    projection = np.fromfile("/home/nv/wyk/Data/ball_sino.raw", dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,128,80).to(device)
    input = np.fromfile("/home/nv/wyk/Data/input.raw", dtype="float32")
    input = torch.from_numpy(input)
    input = input.reshape(512*512,2).to(device)
    theta = torch.autograd.Variable(torch.zeros(24,1)).to(device)
    rotation_matrix = torch.stack([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)])
    translation_matrix = torch.autograd.Variable(torch.zeros(24,2)).to(device)
    # lossFunction = nn.MSELoss()
    lossFunction = lambda x,y: F.mse_loss(x,y) + F.l1_loss(x,torch.zeros_like(x))
    optimizer = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in model]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=200, gamma=0.95) for o in optimizer]
    motion_optimizer = torch.optim.Adam([rotation_matrix, translation_matrix], lr=0.05)
    tic = time.time()
    for e in range(501):
        output = torch.zeros(16, 512*512, 1).to(device)
        for s in range(16):
            output[s,...] = model[s](input).float()
        output_projection = ForwardProjection.apply(output.reshape(1, 1,16,512,512))
        output_projection = translate(output_projection, rotation_matrix, translation_matrix)
        loss = lossFunction(output_projection, projection)
        loss.backward()
        for s in range(16):
            optimizer[s].step()
            optimizer[s].zero_grad()
            scheduler[s].step()
        motion_optimizer.zero_grad()
        motion_optimizer.step()
        if e%100==0:
            output[10,...].detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
            print(f"loss:{loss.item()}")
    output = torch.zeros(16, 512*512, 1).to(device)
    for s in range(16):
        output[s,...] = model[s](input).float()
    print(torch.sum(theta))
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    print(f"time cost:{time.time() - tic}")
