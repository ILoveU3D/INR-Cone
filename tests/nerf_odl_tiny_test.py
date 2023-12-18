###
# Nerf重建二维平行束（hash encoding）
###

import odl
from odl.contrib import torch as odl_torch
import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
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
        return torch.autograd.Variable(residual/(torch.max(residual)+1e-4), requires_grad=True)

encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 8, "n_features_per_level": 8,
    "log2_hash_map_size": 19,
    "base_resolution": 2,
    "per_level_scale": 1.95,
    "interpolation": "Linear"
}
network_config = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "Sigmoid",
    "n_neurons": 64, "n_hidden_layers": 1
}
model = tcnn.NetworkWithInputEncoding(
    n_input_dims=2, n_output_dims=1,
    encoding_config=encoding_config, network_config=network_config
).cuda()


if __name__ == '__main__':
    label = np.fromfile("/media/wyk/wyk/Data/raws/trainData/pa_1.raw", dtype="float32")
    label = np.reshape(label, [64, 256*256])
    label = label[0, ...]
    label /= np.max(label)
    label = torch.from_numpy(label).unsqueeze(-1).cuda()
    label_sino = A(label.reshape(1,256,256))
    label_sino += torch.empty_like(label_sino).normal_()
    # label_sino.detach().cpu().numpy().astype("float32").tofile("/media/wyk/wyk/Data/raws/output.raw")
    input = torch.zeros(256,256,2)
    value = np.linspace(-1,1,256)
    for i in range(256):
        for j in range(256):
                input[i,j,0] = value[i]
                input[i,j,1] = value[j]
    input = input.reshape(256*256,2).cuda()
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    tic = time.time()
    for e in range(2000):
        output = model(input).float()
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
