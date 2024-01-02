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
    def forward(ctx, input):
        result = torch.from_numpy(H * input.cpu().numpy().flatten()).to(device)
        return torch.autograd.Variable(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        # ramp filter (optional)
        grad = torch.fft.ifft2(torch.fft.fft2(grad.reshape(anglesNum, detectorSize)) * ramp).real
        residual = torch.from_numpy(H.T * grad.cpu().numpy().flatten()).to(device)
        return torch.autograd.Variable(residual/(torch.max(residual)+1.0), requires_grad=True)
    
projectorGeometryPerAngle = astra.create_proj_geom('fanflat', 1.0, detectorSize, 0, 800, 400)
projectorPerAngle = astra.create_projector('cuda',projectorGeometryPerAngle,volumeGeometry)
HPerAngle = astra.OpTomo(projectorPerAngle)

class ProjectionPerAngle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        result = torch.from_numpy(HPerAngle * input.cpu().numpy().flatten()).to(device)
        return torch.autograd.Variable(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        # ramp filter (optional)
        grad = torch.fft.ifft2(torch.fft.fft2(grad.reshape(1, detectorSize)) * ramp).real
        residual = torch.from_numpy(HPerAngle.T * grad.cpu().numpy().flatten()).to(device)
        return torch.autograd.Variable(residual/(torch.max(residual)+1.0), requires_grad=True)

angles = np.linspace(np.pi/3, 2*np.pi+np.pi/3, anglesNum, endpoint=False)
projectorGeometryBias = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles, 800, 400)
projectorBias = astra.create_projector('cuda',projectorGeometryBias,volumeGeometry)
HBias = astra.OpTomo(projectorBias)

# Nerf
encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 8, "n_features_per_level": 4,
    "log2_hash_map_size": 19,
    "base_resolution": 2,
    "per_level_scale": 2.0,
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

def build_coordinate(L, angle):
    trans_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]
    )
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    x, y = np.meshgrid(x, y, indexing='ij')  # (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (L*L, 2)
    xy = xy @ trans_matrix.T  # (L*L, 2) @ (2, 2)
    return xy.astype(np.float32)

def build_coordinate_test(L):
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    x, y = np.meshgrid(x, y, indexing='ij')  # (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2).astype(np.float32)  # (L*L, 2)
    return torch.from_numpy(xy).to(device)

class TrainSet(Dataset):
    def __init__(self, angles, volumeSize, sino) -> None:
        super().__init__()
        self.rays = [build_coordinate(volumeSize, angle) for angle in angles]
        self.sino = sino

    def __getitem__(self, index):
        return self.rays[index], self.sino[index]
    
    def __len__(self):
        return len(self.rays)
    
angleCorrection = torch.zeros(angles.shape).to(device)
angleCorrection = torch.autograd.Variable(angleCorrection, requires_grad=True)
angleTranslation = torch.zeros([len(angles), 2]).to(device)
angleTranslation = torch.autograd.Variable(angleTranslation)

def build_translation(grid, idx):
    angle = angleCorrection[idx]
    translation = angleTranslation[idx]
    trans_matrix = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ]
    ).to(device).to(torch.float32)
    return grid @ trans_matrix + translation

if __name__ == '__main__':
    label = np.fromfile("/home/nv/wyk/Data/SheppLogan.raw", dtype="float32")
    label = np.reshape(label, [64, volumeSize[0]*volumeSize[1]])
    label = label[31, ...]
    label_sino = torch.from_numpy(HBias * label).to(device)
    trainSet = TrainSet(angles, volumeSize[0], label_sino.reshape(anglesNum, detectorSize))
    trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
    input = build_coordinate_test(volumeSize[0])
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    optimizer_corr = torch.optim.Adam([angleCorrection, angleTranslation], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_corr, step_size=200, gamma=0.95)
    tic = time.time()
    for e in range(41):
        label_sino_test = np.zeros([anglesNum, detectorSize])
        for i, (grid, label_sino_angle) in enumerate(trainLoader):
            grid = grid.view(-1,2).to(device)
            # grid = build_translation(grid, i)
            label_sino_angle = label_sino_angle.view(detectorSize).to(device)
            if DEBUG_NERF:
                sample_test = F.grid_sample(torch.from_numpy(label).reshape(1, 1, volumeSize[0], volumeSize[1]), grid.cpu().reshape(1,volumeSize[0], volumeSize[1],2))
                sample_test.numpy().tofile(f"/home/nv/wyk/Data/debug/{i}.raw")
            output = model(grid).float().view(-1)
            output_sino = ProjectionPerAngle.apply(output)
            if DEBUG_NERF:
                label_sino_angle_test = ProjectionPerAngle.apply(sample_test)
                label_sino_test[i,:] = label_sino_angle.cpu().numpy()
            loss = lossFunction(output_sino, label_sino_angle)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            # optimizer_corr.step()
            # optimizer_corr.zero_grad()
        if e%10==0:
            print(f"loss:{loss.item()}")
            print(f"angle:{angleCorrection[0]}")
    output = model(input).view(-1)
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    label.astype("float32").tofile("/home/nv/wyk/Data/label.raw")
    print(f"time cost:{time.time() - tic}")
