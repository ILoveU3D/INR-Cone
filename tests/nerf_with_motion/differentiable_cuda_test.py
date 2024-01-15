import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from JITSelfCalibration import differentiableFanFlatGradient as dffg

device = "cuda:0"
sino = torch.eye(360, 900, dtype=torch.float32).to(device)
sino, = torch.gradient(sino, dim=1)
volume = torch.ones(512,512, dtype=torch.float32).to(device)
projectionMatrix = torch.ones(360, 2,3, dtype=torch.float32).to(device)
grad = dffg(sino, volume, projectionMatrix)
# grad = dffg(sino, volume, projectionMatrix)
print(grad.cpu())