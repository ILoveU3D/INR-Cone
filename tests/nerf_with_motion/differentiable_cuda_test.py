import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from JITSelfCalibration import differentiableFanFlatGradient as dffg

device = "cuda:0"
def getProjectionMatrix(angles):
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        return torch.stack([1200*cos_angles, 1200*sin_angles, torch.zeros_like(angles),
                                -sin_angles, cos_angles, 800 * torch.ones_like(angles)]).view(2,3,-1).permute(2,0,1).to(torch.float32).contiguous().to(device)

sino = torch.ones(360, 900, dtype=torch.float32).to(device)
# sino, = torch.gradient(sino, dim=1)
volume = torch.ones(512,512, dtype=torch.float32).to(device)
projectionMatrix = getProjectionMatrix(torch.linspace(0, 2*torch.pi, 360))
grad = dffg(sino, volume, projectionMatrix)
print(grad[0].cpu())
print(grad[180].cpu())