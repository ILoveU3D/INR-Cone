import astra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import time
from JITSelfCalibration import differentiableFanFlatGradient as dffg
DEBUG_NERF = True

def _filter(projWidth):
    filter = np.ones([projWidth], dtype=np.float32)
    mid = np.floor(projWidth / 2)
    for i in range(projWidth):
        filter[i] = mid - np.abs(mid - i)
    return filter

# System matrix & astra module
anglesNum = 360
angles = np.linspace(0, 2*np.pi, anglesNum, endpoint=False)
detectorSize = 900
volumeSize = [512, 512]
projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles, 800, 400)
volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
H = astra.OpTomo(projector)
device = "cuda:2"
ramp = torch.from_numpy(_filter(detectorSize)).to(device)

class Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, angles):
        projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles.tolist(), 800, 400)
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
        projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
        H = astra.OpTomo(projector)
        result = torch.from_numpy(H * input.cpu().numpy().flatten()).to(device)
        ctx.save_for_backward(angles)
        return torch.autograd.Variable(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        # ramp filter (optional)
        angles, = ctx.saved_tensors
        projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles.tolist(), 800, 400)
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
        projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
        H = astra.OpTomo(projector)
        grad = torch.fft.ifft2(torch.fft.fft2(grad.reshape(anglesNum, detectorSize)) * ramp).real
        residual = torch.from_numpy(H.T * grad.cpu().numpy().flatten()).to(device)
        residual /= torch.max(residual)+1.0
        return torch.autograd.Variable(residual, requires_grad=True), None

class ProjectionGeom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, projectionMatrix, label):
        angles = ProjectionGeom.getResultAngle(projectionMatrix)
        projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles.tolist(), 800, 400)
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
        projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
        H = astra.OpTomo(projector)
        result = torch.from_numpy(H * input.cpu().numpy().flatten()).to(device)
        residual = label - input
        ctx.save_for_backward(projectionMatrix, result, residual)
        return torch.autograd.Variable(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        # ramp filter (optional)
        projectionMatrix, sino, residual = ctx.saved_tensors
        gsino, = torch.gradient(sino.reshape(anglesNum, detectorSize), dim=1)
        gMatrix = dffg(gsino, residual, projectionMatrix.contiguous())
        gMatrix[11:,...] = 0
        return None, gMatrix, None
    
    @staticmethod
    def getResultAngle(projectionMatrix):
        return torch.atan2(-projectionMatrix[:, 1, 0], projectionMatrix[:, 1, 1]).to(torch.float32).to(device)

anglesBias = angles.copy()
anglesBias[0:10] += np.pi/10
projectorGeometryBias = astra.create_proj_geom('fanflat', 1.0, detectorSize, anglesBias, 800, 400)
projectorBias = astra.create_projector('cuda',projectorGeometryBias,volumeGeometry)
HBias = astra.OpTomo(projectorBias)

def getProjectionMatrix(angles):
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    return torch.stack([1200*cos_angles, 1200*sin_angles, torch.zeros_like(angles),
                            -sin_angles, cos_angles, 800 * torch.ones_like(angles)]).view(2,3,-1).permute(2,0,1).to(torch.float32).to(device)

def constuctProjectionMatrix(angles, trans):
    return torch.stack([torch.cos(angles), -torch.sin(angles), torch.ones_like(angles) * trans[:,0],
                        torch.sin(angles), torch.cos(angles), torch.ones_like(angles) * trans[:,1],
                        torch.zeros_like(angles), torch.zeros_like(angles), torch.ones_like(angles)]).view(3,3,-1).permute(2,0,1).to(torch.float32).to(device)

import matplotlib.pyplot as plt
def drawAngles(angles, path, label=None):
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    if label is not None:
        ax.plot(label, np.ones_like(label), 'o')
    ax.plot(angles, np.ones_like(angles), '.')
    ax.set_axis_off()
    plt.savefig(path)
    plt.close('all')

# Nerf
encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 8, "n_features_per_level": 8,
    "log2_hash_map_size": 22,
    "base_resolution": 3,
    "per_level_scale": 1.95,
    "interpolation": "Linear"
}
network_config = {
    "otype": "CutlassMLP",
    "activation": "ReLU",
    "output_activation": "ReLU",
    "n_neurons": 64, "n_hidden_layers": 1
}
model = tcnn.NetworkWithInputEncoding(
    n_input_dims=2, n_output_dims=1,
    encoding_config=encoding_config, network_config=network_config
).to(device)

# dataSet per angle
from torch.utils.data import Dataset, DataLoader

def build_coordinate_test(L):
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    x, y = np.meshgrid(x, y, indexing='ij')  # (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2).astype(np.float32)  # (L*L, 2)
    return torch.from_numpy(xy).to(device)

if __name__ == '__main__':
    label = np.fromfile("/home/nv/wyk/Data/star.raw", dtype="float32")
    # label = np.reshape(label, [64, volumeSize[0]*volumeSize[1]])
    # label = label[32, ...]
    label_sino = torch.from_numpy(HBias * label).to(device)
    input = build_coordinate_test(volumeSize[0])
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.95)
    anglesCorr = torch.from_numpy(angles).to(device)
    # anglesCorr = torch.autograd.Variable(anglesCorr, requires_grad=True)
    # optimizerCorr = torch.optim.Adam([anglesCorr], lr=0.05)
    projectionMatrixInit = getProjectionMatrix(anglesCorr)
    projectionMatrixCorrAngle = torch.autograd.Variable(torch.tensor([0.0] * anglesNum).to(device), requires_grad=True)
    projectionMatrixCorrTrans = torch.autograd.Variable(torch.tensor([[0.0,0.0]] * anglesNum).to(device), requires_grad=True)
    optimizerCorr = torch.optim.Adam([projectionMatrixCorrAngle, projectionMatrixCorrTrans], lr=1e-2)
    label = torch.from_numpy(label).to(device)
    tic = time.time()
    for e in range(501):
        output = model(input).float().view(-1)
        output_sino = Projection.apply(output, anglesCorr)
        loss = lossFunction(output_sino, label_sino)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if e%100==0:
            print(f"loss:{loss.item()}")
    for iph in range(0):
        for e in range(101):
            output = model(input).float().view(-1)
            output_sino = Projection.apply(output, anglesCorr)
            loss = lossFunction(output_sino, label_sino)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if e%100==0:
                print(f"loss:{loss.item()}")
        print("geometry finetune...")
        output = model(input).float().view(-1)
        projectionMatrixCorr = constuctProjectionMatrix(projectionMatrixCorrAngle, projectionMatrixCorrTrans)
        projectionMatrixCorr = projectionMatrixCorr.reshape(anglesNum, 3, 3).to(device)
        projectionMatrix = projectionMatrixInit @ projectionMatrixCorr
        output_sino = ProjectionGeom.apply(output, projectionMatrix, label)
        loss = lossFunction(output_sino, label_sino)
        loss.backward()
        optimizerCorr.step()
        optimizerCorr.zero_grad()
        anglesCorr = ProjectionGeom.getResultAngle(projectionMatrix.detach())
        print(f"--> geometry loss:{np.mean(np.square(anglesCorr.detach().cpu().numpy()-anglesBias))}")
    output = model(input).float().view(-1)
    drawAngles(anglesCorr.detach().cpu().numpy()[::5], "/home/nv/wyk/Data/geo/angles.jpg", anglesBias[::5])
    drawAngles(anglesCorr.detach().cpu().numpy()[:10], "/home/nv/wyk/Data/geo/angles02.jpg", anglesBias[:10])
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    label.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/label.raw")
    print(f"time cost:{time.time() - tic}")