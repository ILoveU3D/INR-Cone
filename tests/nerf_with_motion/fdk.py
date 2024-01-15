import numpy as np
import torch
import os
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from ConeBeamLayers.BeijingGeometry import ForwardProjection
device = "cuda:3"

if __name__ == '__main__':
    projection = np.fromfile("/home/nv/wyk/Data/lz/projection2.raw", dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,144,80).to(device)
    projection[torch.isnan(projection)] = 0
    lossFunction = torch.nn.MSELoss()
    output = torch.autograd.Variable(torch.zeros(64, 512, 512, dtype=torch.float32).to(device), requires_grad=True)
    output_projection = ForwardProjection.apply(output.reshape(1, 1,64,512,512))
    loss = lossFunction(projection, output_projection)
    loss.backward()
    print(torch.where(torch.isnan(output.grad)))
    (-output.grad.cpu().numpy()).astype("float32").tofile("/home/nv/wyk/Data/lz/output.raw")
