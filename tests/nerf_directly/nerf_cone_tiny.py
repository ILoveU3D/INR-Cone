import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
import time
import os
from pytorch_ssim import ssim
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from ConeBeamLayers.BeijingGeometry import ForwardProjection
from model import model

device = "cuda:3"
for m in model:
    m = m.to(device)
    m.train()

if __name__ == '__main__':
    # label = np.fromfile("/home/nv/wyk/Data/balls.raw", dtype="float32")
    # label = np.reshape(label, [1, 1, 16, 512, 512])
    # label = label[:,:, 24:40, ...]
    # label = torch.from_numpy(label).to(device)
    # label /= np.max(label)
    # projection = ForwardProjection.apply(label)
    # label = torch.from_numpy(label).view(-1).unsqueeze(-1).cuda()
    projection = np.fromfile("/home/nv/wyk/Data/ball_sino.raw", dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,128,80).to(device)
    input = np.fromfile("/home/nv/wyk/Data/input.raw", dtype="float32")
    input = torch.from_numpy(input)
    input = input.reshape(512*512,2).to(device)
    lossFunction = nn.MSELoss()
    # lossFunction = lambda x,y: torch.nn.functional.mse_loss(x,y) + 100*(1 - ssim(x.squeeze(0), y.squeeze(0)))
    optimizer = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in model]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=200, gamma=0.95) for o in optimizer]
    tic = time.time()
    for e in range(500):
        output = torch.zeros(16, 512*512, 1).to(device)
        for s in range(16):
            output[s,...] = model[s](input).float()
        output_projection = ForwardProjection.apply(output.reshape(1, 1,16,512,512))
        loss = lossFunction(output_projection, projection)
        loss.backward()
        for s in range(16):
            optimizer[s].step()
            optimizer[s].zero_grad()
            scheduler[s].step()
        if e%100==0:
            output[10,...].detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
            print(f"loss:{loss.item()}")
    output = torch.zeros(16, 512*512, 1).to(device)
    for s in range(16):
        output[s,...] = model[s](input).float()
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    # label.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/label.raw")
    torch.save(model.state_dict(), "/home/nv/wyk/Data/nerf_ball.pth")
    print(f"time cost:{time.time() - tic}")
