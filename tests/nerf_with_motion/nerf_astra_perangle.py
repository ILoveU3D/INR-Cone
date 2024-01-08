import astra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import time
DEBUG_NERF = False

def _filter(projWidth):
    filter = np.ones([projWidth], dtype=np.float32)
    mid = np.floor(projWidth / 2)
    for i in range(projWidth):
        filter[i] = mid - np.abs(mid - i)
    return filter

# System matrix & astra module
anglesNum = 45
angles = np.linspace(0, 2*np.pi, anglesNum, endpoint=False)
detectorSize = 900
volumeSize = [512, 512]
projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles, 800, 400)
volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
H = astra.OpTomo(projector)
device = "cuda:2"
ramp = torch.from_numpy(_filter(detectorSize)).to(device)
    
class ProjectionPerAngle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, angle):
        projectorGeometryPerAngle = astra.create_proj_geom('fanflat', 1.0, detectorSize, angle.item(), 800, 400)
        projectorPerAngle = astra.create_projector('cuda',projectorGeometryPerAngle,volumeGeometry)
        HPerAngle = astra.OpTomo(projectorPerAngle)
        result = torch.from_numpy(HPerAngle * input.cpu().numpy().flatten()).to(device)
        ctx.save_for_backward(angle)
        return torch.autograd.Variable(result, requires_grad=True)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        # ramp filter (optional)
        angle, = ctx.saved_tensors
        projectorGeometryPerAngle = astra.create_proj_geom('fanflat', 1.0, detectorSize, angle.item(), 800, 400)
        projectorPerAngle = astra.create_projector('cuda',projectorGeometryPerAngle,volumeGeometry)
        HPerAngle = astra.OpTomo(projectorPerAngle)
        grad = torch.fft.ifft2(torch.fft.fft2(grad.reshape(1, detectorSize)) * ramp).real
        residual = torch.from_numpy(HPerAngle.T * grad.cpu().numpy().flatten()).to(device)
        return torch.autograd.Variable(residual/(torch.max(residual)+1.0), requires_grad=True), torch.IntTensor(0)

# Nerf
encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 8, "n_features_per_level": 8,
    "log2_hash_map_size": 22,
    "base_resolution": 2,
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
    label = np.fromfile("/home/nv/wyk/Data/SheppLogan.raw", dtype="float32")
    label = np.reshape(label, [64, volumeSize[0]*volumeSize[1]])
    label = label[32, ...]
    label /= np.max(label)
    label_sino = torch.from_numpy(H * label).to(device)
    trainSet = TrainSet(angles, label_sino.reshape(anglesNum, detectorSize))
    trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
    input = build_coordinate_test(volumeSize[0])
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    tic = time.time()
    for e in range(91):
        meanLoss = 0
        for i, (angle, label_sino_angle) in enumerate(trainLoader):
            label_sino_angle = label_sino_angle.view(detectorSize).to(device)
            output = model(input).float().view(-1)
            output_sino = ProjectionPerAngle.apply(output, angle)
            loss = lossFunction(output_sino, label_sino_angle)
            meanLoss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        if e%10==0:
            print(f"loss:{meanLoss/anglesNum}")
    output = model(input).view(-1)
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    label.astype("float32").tofile("/home/nv/wyk/Data/label.raw")
    print(f"time cost:{time.time() - tic}")