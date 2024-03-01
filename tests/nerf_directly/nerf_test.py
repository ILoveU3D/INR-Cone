import numpy as np
import torch
import torch.nn as nn
import time
L = 30

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
    label = np.fromfile("/media/wyk/wyk/Data/raws/trainData/pa_436.raw", dtype="float32")
    label = np.reshape(label, [64, 256*256])
    label = label[63, ...]
    label /= np.max(label)
    label = torch.from_numpy(label).cuda()
    input = torch.zeros(256,256,L,2,2)
    value = np.linspace(-1,1,256)
    for i in range(256):
        for j in range(256):
            for l in range(L):
                input[i,j,l,0,0] = np.sin(np.power(2, l) * np.pi * value[i])
                input[i,j,l,0,1] = np.sin(np.power(2, l) * np.pi * value[j])
                input[i,j,l,1,0] = np.cos(np.power(2, l) * np.pi * value[i])
                input[i,j,l,1,1] = np.cos(np.power(2, l) * np.pi * value[j])
    input = input.reshape(256*256,2*2*L).cuda()
    model = NeRF().cuda()
    lossFunction = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.95)
    tic = time.time()
    for e in range(30000):
        output = model(input)
        loss = lossFunction(label, output)
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