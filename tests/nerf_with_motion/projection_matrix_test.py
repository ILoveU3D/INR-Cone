import numpy as np
import torch
from JITSelfCalibration import backward, forward
import astra

device = "cuda:0"
anglesNum = 360
angles = np.linspace(0, 2*np.pi, anglesNum, endpoint=False)
detectorSize = 900
volumeSize = [512, 512]
projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles, 800, 400)
volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
H = astra.OpTomo(projector)

def getProjectionMatrix(angles):
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    return torch.stack([1200*cos_angles, 1200*sin_angles, torch.zeros_like(angles),
                            -sin_angles, cos_angles, 800 * torch.ones_like(angles)]).view(2,3,-1).permute(2,0,1).to(torch.float32).to(device).contiguous()

label = np.fromfile("/home/nv/wyk/Data/balls.raw", dtype="float32")
label = np.reshape(label, [16, volumeSize[0]*volumeSize[1]])
label = label[11, ...]
label_sino = torch.from_numpy(H * label.flatten()).to(device)
label_sino.cpu().numpy().tofile("/home/nv/wyk/Data/label_sino.raw")
projectionMatrix = getProjectionMatrix(torch.from_numpy(angles))
volume = backward(label_sino.reshape(anglesNum, detectorSize), projectionMatrix)
volume.cpu().numpy().tofile("/home/nv/wyk/Data/output.raw")
sino = forward(torch.from_numpy(label).to(device), projectionMatrix)
sino.cpu().numpy().tofile("/home/nv/wyk/Data/sino.raw")