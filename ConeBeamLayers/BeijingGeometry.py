import numpy as np
import torch
import scipy.io as sco
import JITBeijingGeometry as projector
from options import *

def __cosWeight__():
    cosine = np.zeros(
        [1, 1, beijingAngleNum, beijingSubDetectorSize[1], (beijingSubDetectorSize[0] + beijingGap) * beijingPlanes],
        dtype=np.float32)
    mid = np.array([beijingSubDetectorSize[1], (beijingSubDetectorSize[0] + beijingGap) * beijingPlanes]) / 2
    for i in range(beijingSubDetectorSize[1]):
        for j in range((beijingSubDetectorSize[0] + beijingGap) * beijingPlanes):
            cosine[..., i, j] = beijingSDD / np.sqrt(beijingSDD ** 2 + (i - mid[1]) ** 2 + (j - mid[0]) ** 2)
    return torch.from_numpy(cosine)

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
cosWeight = torch.nn.Parameter(__cosWeight__(), requires_grad=False)
ramp = torch.nn.Parameter(__conv__((beijingSubDetectorSize[0] + beijingGap) * beijingPlanes), requires_grad=False)

class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        sino = projector.forward(input, volumeSize.to(device), detectorSize.to(device), parameters.to(device), device.index) * sampleInterval
        return sino.reshape(beijingAngleNum, beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        grad = grad.view(1, 1, beijingAngleNum, beijingSubDetectorSize[1], (beijingSubDetectorSize[0] + beijingGap) * beijingPlanes)
        grad *= cosWeight.to(device)
        grad = grad.view(1, 1, beijingAngleNum * beijingSubDetectorSize[1], (beijingSubDetectorSize[0] + beijingGap) * beijingPlanes)
        grad = torch.nn.functional.conv2d(grad, ramp.to(device), stride=1, padding=(0, int((beijingSubDetectorSize[0] + beijingGap) * beijingPlanes / 2)))
        grad = grad.reshape(beijingAngleNum, beijingSubDetectorSize[1], beijingPlanes, beijingSubDetectorSize[0]).permute(0,2,1,3).reshape(1, 1, beijingAngleNum*beijingPlanes, beijingSubDetectorSize[1], beijingSubDetectorSize[0])
        volume = projector.backward(grad, volumeSize.to(device), detectorSize.to(device), parameters.to(device), beijingGap, device.index)
        return volume / (torch.max(volume.view(-1))+1e-6)
