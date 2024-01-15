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
    def forward(ctx, input, angles):
        projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles.tolist(), 800, 400)
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
        projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
        H = astra.OpTomo(projector)
        result = torch.from_numpy(H * input.cpu().numpy().flatten()).to(device)
        projectionMatrix = ProjectionGeom.getProjectionMatrix(angles)
        ctx.save_for_backward(projectionMatrix, result)
        return torch.autograd.Variable(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        # ramp filter (optional)
        projectionMatrix, sino = ctx.saved_tensors
        angles = ProjectionGeom.getResultAngle(projectionMatrix)
        projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles.tolist(), 800, 400)
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
        projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
        H = astra.OpTomo(projector)
        grad = torch.fft.ifft2(torch.fft.fft2(grad.reshape(anglesNum, detectorSize)) * ramp).real
        residual = torch.from_numpy(H.T * grad.cpu().numpy().flatten()).to(device)
        residual /= torch.max(residual)+1.0
        gsino, = torch.gradient(sino.reshape(anglesNum, detectorSize), dim=1)
        gMatrix = dffg(torch.ones_like(gsino), residual, projectionMatrix.contiguous(), torch.ones_like(grad))
        gangle = ProjectionGeom.getResultAngle(gMatrix)
        return None, gangle
    
    @staticmethod
    def getProjectionMatrix(angles):
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        return torch.stack([1200*cos_angles, 1200*sin_angles, torch.zeros_like(angles),
                                -sin_angles, cos_angles, 800 * torch.ones_like(angles)]).view(2,3,-1).permute(2,0,1).to(torch.float32).to(device)
    
    @staticmethod
    def getResultAngle(projectionMatrix):
        return torch.atan2(-projectionMatrix[:, 1, 0], projectionMatrix[:, 1, 1]).to(torch.float32).to(device)

anglesBias = angles.copy()
# anglesBias[0:10] += np.pi/10
projectorGeometryBias = astra.create_proj_geom('fanflat', 1.0, detectorSize, anglesBias, 800, 400)
projectorBias = astra.create_projector('cuda',projectorGeometryBias,volumeGeometry)
HBias = astra.OpTomo(projectorBias)

import matplotlib.pyplot as plt
def drawAngles(angles, path, label=None):
    angles = angles[::5]
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    ax.plot(angles, np.ones_like(angles), '.')
    if label is not None:
        label = label[::5]
        ax.plot(label, np.ones_like(label), '.')
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
    "per_level_scale": 1.9,
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

class TrainSet(Dataset):
    def __init__(self, angles, sino) -> None:
        super().__init__()
        self.angles = angles
        self.sino = sino

    def __getitem__(self, index):
        return self.angles[index], self.sino[index]
    
    def __len__(self):
        return len(self.angles)

if __name__ == '__main__':
    label = np.fromfile("/home/nv/wyk/Data/balls.raw", dtype="float32")
    label = np.reshape(label, [16, volumeSize[0]*volumeSize[1]])
    label = label[11, ...]
    label_sino = torch.from_numpy(HBias * label).to(device)
    input = build_coordinate_test(volumeSize[0])
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.95)
    angles = torch.from_numpy(angles).to(device)
    anglesCorr = torch.autograd.Variable(torch.zeros_like(angles).to(device), requires_grad=True)
    optimizerCorr = torch.optim.Adam([anglesCorr], lr=1e-3)
    tic = time.time()
    for ite in range(1):
        for e in range(500):
            output = model(input).float().view(-1)
            output_sino = Projection.apply(output, angles)
            loss = lossFunction(output_sino, label_sino)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if e%100==0:
                print(f"loss:{loss.item()}")
        print("geometry finetune...")
        output = model(input).float().view(-1)
        for e in range(20):
            anglesNew = angles + anglesCorr
            output_sino = ProjectionGeom.apply(output, anglesNew)
            loss = lossFunction(output_sino, label_sino)
            loss.backward()
            optimizerCorr.step()
            optimizerCorr.zero_grad()
            if e%10==0:
                print(f"geometry loss:{np.mean(np.square(anglesNew.detach().cpu().numpy()-anglesBias))}")
        angles += anglesCorr.detach()
    drawAngles(angles.detach().cpu().numpy(), "/home/nv/wyk/Data/geo/angles.jpg", anglesBias)
    output = model(input).view(-1)
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    label.astype("float32").tofile("/home/nv/wyk/Data/label.raw")
    print(f"time cost:{time.time() - tic}")