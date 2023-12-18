###
# Nerf重建二维平行束（w/o position encoding）
###

import odl
from odl.contrib import torch as odl_torch
import numpy as np
import torch
import torch.nn as nn
import time

detector_size = 500
volume_size = 256
space = odl.uniform_discr([-1,-1], [1,1], [volume_size, volume_size], dtype="float32")
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=360, det_shape=detector_size)
operator = odl.tomo.RayTransform(space, geometry)
A = odl_torch.OperatorAsModule(operator)
AT = odl_torch.OperatorAsModule(odl.tomo.fbp_op(operator))
class Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.autograd.Variable(A(input), requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        residual = AT(grad)
        return torch.autograd.Variable(residual/torch.max(residual), requires_grad=True)

L = 15
class NeRF(nn.Module):
    def __init__(self, features=256):
        super(NeRF, self).__init__()
        self.renderer = nn.Sequential(
            nn.Linear(2*2*L, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 1),
            nn.Sigmoid()
        )
        def kaiming_init(model):
            if isinstance(model, nn.Linear):
                nn.init.kaiming_uniform_(model.weight)
                nn.init.zeros_(model.bias)
        self.apply(kaiming_init)

    def forward(self, x):
        return self.renderer(x).squeeze(-1)

if __name__ == '__main__':
    label = np.fromfile("/media/wyk/wyk/Data/raws/trainData/pa_1.raw", dtype="float32")
    label = np.reshape(label, [64, 256*256])
    label = label[0, ...]
    label /= np.max(label)
    label = torch.from_numpy(label).cuda()
    label_sino = A(label.reshape(1,256,256))
    input = torch.zeros(256,256,L,2,2)
    # input = torch.zeros(256, 256, 2)
    value = np.linspace(-1,1,256)
    for i in range(256):
        for j in range(256):
            for l in range(L):
                input[i,j,l,0,0] = np.sin(np.power(2, l) * np.pi * value[i])
                input[i,j,l,0,1] = np.sin(np.power(2, l) * np.pi * value[j])
                input[i,j,l,1,0] = np.cos(np.power(2, l) * np.pi * value[i])
                input[i,j,l,1,1] = np.cos(np.power(2, l) * np.pi * value[j])
            # input[i, j, 0] = value[i]
            # input[i, j, 1] = value[j]
    input = input.reshape(256*256,2*2*L).cuda()
    # input = input.reshape(256 * 256, 2).cuda()
    model = NeRF().cuda()
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
    tic = time.time()
    for e in range(10000):
        output = model(input)
        output_sino = Projection.apply(output.reshape(1,256,256))
        loss = lossFunction(output_sino, label_sino)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if e%100==0:
            print(f"loss:{loss.item()}")
    output = model(input)
    output.detach().cpu().numpy().astype("float32").tofile("/media/wyk/wyk/Data/raws/output.raw")
    label.detach().cpu().numpy().astype("float32").tofile("/media/wyk/wyk/Data/raws/label.raw")
    print(f"time cost:{time.time() - tic}")