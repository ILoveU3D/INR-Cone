import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
import time
import os
os.chdir("/media/wyk/wyk/Recon/inf-recon")
from ConeBeamLayers.BeijingGeometry import ForwardProjection

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
    n_input_dims=3, n_output_dims=1,
    encoding_config=encoding_config, network_config=network_config
).cuda()


if __name__ == '__main__':
    label = np.fromfile("/media/wyk/wyk/Data/raws/SheppLogan.raw", dtype="float32")
    label = np.reshape(label, [1, 1, 64, 512, 512])
    label = label[:, 24:40, ...]
    label = torch.from_numpy(label).cuda()
    # label /= np.max(label)
    projection = ForwardProjection.apply(label)
    # label = torch.from_numpy(label).view(-1).unsqueeze(-1).cuda()
    input = np.fromfile("/media/wyk/wyk/Data/raws/input.raw", dtype="float32")
    input = torch.from_numpy(input)
    input = input.reshape(512*512*16,3).cuda()
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
    tic = time.time()
    for e in range(2000):
        output = model(input).float()
        output_projection = ForwardProjection.apply(output.reshape(1, 1,16,512,512))
        loss = lossFunction(output_projection, projection)
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
