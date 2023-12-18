###
# tiny testing
###

import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
import time

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
device = "cuda:3"
model = tcnn.NetworkWithInputEncoding(
    n_input_dims=2, n_output_dims=1,
    encoding_config=encoding_config, network_config=network_config
).to(device)


if __name__ == '__main__':
    label = np.fromfile("/home/nv/wyk/Data/label.raw", dtype="float32")
    label = np.reshape(label, [256*256])
    label /= np.max(label)
    label = torch.from_numpy(label).unsqueeze(-1).to(device)
    # label_sino += torch.empty_like(label_sino).normal_()
    input = torch.zeros(256,256,2)
    value = np.linspace(-1,1,256)
    for i in range(256):
        for j in range(256):
                input[i,j,0] = value[i]
                input[i,j,1] = value[j]
    input = input.reshape(256*256,2).to(device)
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    tic = time.time()
    for e in range(20000):
        output = model(input).float()
        loss = lossFunction(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if e%100==0:
            print(f"loss:{loss.item()}")
    output = model(input)
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    print(f"time cost:{time.time() - tic}")
