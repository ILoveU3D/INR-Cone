import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from JITSelfCalibration import differentiableConeGradient as dffg

device = "cuda:0"
sino = torch.rand(360, 288, 160, dtype=torch.float32).to(device)
sinoX, = torch.gradient(sino, dim=1)
sinoY, = torch.gradient(sino, dim=2)
volume = torch.ones(512,512,16, dtype=torch.float32).to(device)
projectionMatrix = torch.ones(360, 3, 4, dtype=torch.float32).to(device)
grad = dffg(sinoX, sinoY, volume, projectionMatrix)
print(grad[0].cpu())
print(grad[180].cpu())