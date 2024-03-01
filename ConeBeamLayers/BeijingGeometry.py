import numpy as np
import torch
import scipy.io as sco
import JITBeijingGeometry as projector
from options import *

def _filter(projWidth):
    filter = np.ones([projWidth], dtype=np.float32)
    mid = np.floor(projWidth / 2)
    for i in range(projWidth):
        filter[i] = mid - np.abs(mid - i)
    return filter

def __conv__(projWidth):
    filter = np.ones([1, 1, 1, projWidth + 1], dtype=np.float32)
    mid = np.floor(projWidth / 2)
    for i in range(projWidth + 1):
        if (i - mid) % 2 == 0:
            filter[..., i] = 0
        else:
            filter[..., i] = -0.5 / (np.pi * np.pi * (i - mid) * (i - mid))
        if i == mid:
            filter[..., i] = 1 / 8
    return torch.from_numpy(filter)

parameters = sco.loadmat(beijingParameterRoot)
parameters = np.array(parameters["projection_matrix"]).astype(np.float32)
parameters = torch.from_numpy(parameters).contiguous()
volumeSize = torch.IntTensor(beijingVolumeSize)
detectorSize = torch.IntTensor(beijingSubDetectorSize)
ramp = torch.nn.Parameter(__conv__(beijingSubDetectorSize[0] * beijingPlanes), requires_grad=False)
rampfft = torch.from_numpy(_filter(beijingSubDetectorSize[0] * beijingPlanes))

class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        sino = projector.forward(input, volumeSize.to(device), detectorSize.to(device), parameters.to(device), device.index)
        return sino.reshape(beijingAngleNum, beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        grad = grad.view(1, 1, beijingAngleNum * beijingSubDetectorSize[1], (beijingSubDetectorSize[0]) * beijingPlanes)
        grad = torch.fft.ifft2(torch.fft.fft2(grad) * rampfft.to(device)).real
        # grad = torch.nn.functional.conv2d(grad, ramp.to(device), stride=1, padding=(0, int((beijingSubDetectorSize[0]) * beijingPlanes / 2)))
        grad = grad.reshape(beijingAngleNum, beijingSubDetectorSize[1], beijingPlanes, beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])
        volume = projector.backward(grad, volumeSize.to(device), detectorSize.to(device), parameters.to(device), device.index)
        return volume / (torch.max(volume.view(-1))+1e-6)
