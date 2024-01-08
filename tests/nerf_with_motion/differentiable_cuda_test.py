import torch
from JITSelfCalibration import differentiableFanFlatGradient as dffg

device = "cuda:0"
sino = torch.ones(900, dtype=torch.float32).to(device)
sino, = torch.gradient(sino)
volume = torch.ones(512,512, dtype=torch.float32).to(device)
projectionMatrix = torch.ones(2,3, dtype=torch.float32).to(device)
grad = dffg(sino, volume, projectionMatrix)
print(grad.cpu())