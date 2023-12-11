import numpy as np
import torch
import torch.nn as nn
import time
from model import model

device = "cuda:1"
lossFunction = nn.MSELoss()
optimizer = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in model]
scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=200, gamma=0.95) for o in optimizer]
for m in model:
    m = m.to(device)
    m.train()

if __name__ == '__main__':
    label = np.fromfile("/home/nv/wyk/Data/SheppLogan.raw", dtype="float32")
    label = np.reshape(label, [64, 512*512])
    label = label[:, ...]
    label = torch.from_numpy(label).view(64, -1).unsqueeze(-1).to(device)
    coordinate = np.fromfile("/home/nv/wyk/Data/input.raw", dtype="float32")
    coordinate = torch.from_numpy(coordinate).reshape(-1,2).to(device)
    # label /= np.max(label)
    # label = torch.from_numpy(label).unsqueeze(-1).cuda()
    # label_sino = A(label.reshape(1,256,256))
    # label_sino += torch.empty_like(label_sino).normal_()
    # label_sino.detach().cpu().numpy().astype("float32").tofile("/media/wyk/wyk/Data/raws/output.raw")
    tic = time.time()
    for e in range(2000):
        for s in range(64):
            meanLoss = 0
            output = model[s](coordinate).float()
            loss = lossFunction(output, label[s,...])
            loss.backward()
            optimizer[s].step()
            optimizer[s].zero_grad()
            scheduler[s].step()
            meanLoss += loss.item()
        if e%100==0: print(f"epoch:{e}, loss:{meanLoss/16}")
    output = torch.zeros(64, 512*512, 1).to(device)
    for s in range(64):
        output[s,...] = model[s](coordinate).float()
    output.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/output.raw")
    label.detach().cpu().numpy().astype("float32").tofile("/home/nv/wyk/Data/label.raw")
    print(f"time cost:{time.time() - tic}")
