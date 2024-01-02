import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import time
import os
from pytorch_ssim import ssim
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from ConeBeamLayers.BeijingGeometry import ForwardProjection
from model import model
from options import beijingVolumeSize
from pytorch_ssim import ssim

device = "cuda:2"
model = model.to(device)
model.train()

network_config = {
    "otype": "CutlassMLP",
    "activation": "ReLU",
    "output_activation": "Sigmoid",
    "n_neurons": 64, "n_hidden_layers": 3
}
stl = torch.nn.ModuleList()
for s in range(beijingVolumeSize[2]):
    stl.append(tcnn.Network(
        n_input_dims=2, n_output_dims=2,
        network_config=network_config
    ))
stl = stl.to(device)
stl.train()

def sampleIt(image, grid):
    return F.grid_sample(image.view(1,1,512,512), grid.view(1,512,512,2)).view(-1,1)

if __name__ == '__main__':
    projection = np.fromfile("/home/nv/wyk/Data/ball_sino.raw", dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,128,80).to(device)
    label = np.fromfile("/home/nv/wyk/Data/balls.raw", dtype="float32")
    label = torch.from_numpy(label).reshape(16,512*512,1).to(device)
    label_sino = ForwardProjection.apply(label.reshape(1, 1,beijingVolumeSize[2],512,512))
    input = np.fromfile("/home/nv/wyk/Data/input.raw", dtype="float32")
    input = torch.from_numpy(input)
    input = input.reshape(512*512,2).to(device)
    lossFunction = nn.MSELoss()
    # lossFunction = lambda x,y: ssim(x.view(1,16,512,512), y.view(1,16,512,512))
    optimizer = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in model]
    optimizer_pos = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in stl]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=300, gamma=0.95) for o in optimizer]
    scheduler_pos = [torch.optim.lr_scheduler.StepLR(o, step_size=300, gamma=0.95) for o in optimizer_pos]
    tic = time.time()
    for e in range(1000):
        output = torch.zeros(beijingVolumeSize[2], 512*512, 1).to(device)
        for s in range(beijingVolumeSize[2]):
            calibration = stl[s](input).int()+input
            output[s,...] = model[s](calibration).float()
        output_projection = ForwardProjection.apply(output.reshape(1, 1,beijingVolumeSize[2],512,512))
        loss = lossFunction(output_projection, projection)# + lossFunction(output_projection, label_sino)
        loss.backward(retain_graph=True)
        for s in range(beijingVolumeSize[2]):
            optimizer[s].step()
            optimizer[s].zero_grad()
            scheduler[s].step()
        loss_pos = lossFunction(output_projection, label_sino)
        loss_pos.backward()
        for s in range(beijingVolumeSize[2]):
            optimizer_pos[s].step()
            optimizer_pos[s].zero_grad()
            scheduler_pos[s].step()
        if e%100==0:
            output[8,...].detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
            print(f"loss:{loss.item()}")
    output = torch.zeros(beijingVolumeSize[2], 512*512, 1).to(device)
    for s in range(beijingVolumeSize[2]):
        calibration = stl[s](input).int()+input
        output[s,...] = model[s](calibration).float()
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    # label.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/label.raw")
    print(f"time cost:{time.time() - tic}")
