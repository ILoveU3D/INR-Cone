# It is still in BETA!
# By Wangyukang Mar 1, 2024

import numpy as np
import torch
import torch.nn as nn
import tqdm
import time
from JITSelfCalibration import differentiableConeGradient as dffg
import ConeBeamLayers.BeijingGeometry as geometry
from ConeBeamLayers.BeijingGeometry import ForwardProjection
from model import model
from options import beijingVolumeSize
from options import beijingAngleNum, beijingPlanes
DEBUG_NERF = True

device = "cuda:2"
for m in model:
    m = m.to(device)
    m.train()

def getProjectionMatrix(projectionMatrix):
    s,d,v,u = projectionMatrix[:,0:3], projectionMatrix[:,3:6], projectionMatrix[:,6:9], projectionMatrix[:,9:12]
    posTrans = lambda x: x[:,[1,2,0]]
    normU = torch.norm(u, dim=1).unsqueeze(-1)
    normV = torch.norm(v, dim=1).unsqueeze(-1)
    s = posTrans(s)
    d = posTrans(d)
    v = posTrans(v) / normV
    u = posTrans(u) / normU
    sod = torch.cross(u, v, dim=1)
    sod /= torch.norm(sod, dim=1, keepdim=True)
    mr = torch.inverse(torch.stack((u, v, sod), dim=2))
    vt = torch.matmul(-mr, s.unsqueeze(-1)).squeeze(-1)
    mrt = torch.cat((mr, vt.unsqueeze(-1)), dim=2)
    ktemp = torch.matmul(mr, d.unsqueeze(-1)).squeeze(-1) + vt
    mk = torch.zeros_like(mr)
    mk[:,0,2] = -ktemp[:,0]
    mk[:,1,2] = -ktemp[:,1]
    mk[:,0,0] = ktemp[:,2]
    mk[:,1,1] = ktemp[:,2]
    mk[:,2,2] = 1
    return mk @ mrt, normU, normV

def getResultParams(projectionMatrix, normU, normV):
    N = projectionMatrix.shape[0]
    _, r = torch.linalg.qr(torch.inverse(projectionMatrix[:,:,:3]), mode="r")
    mk = torch.inverse(r)
    mk /= mk[:,2,2].clone().unsqueeze(-1).unsqueeze(-1)
    mk[:,:2,:] *= -1
    mrt = torch.inverse(mk) @ projectionMatrix
    s = torch.linalg.solve(mrt[:,:,:3], -mrt[:,:,3])
    d = torch.linalg.solve(mrt[:,:,:3], torch.stack((-mk[:,0,2], -mk[:,1,2], mk[:,0,0]), dim=1)-mrt[:,:,3])
    u = torch.linalg.solve(mrt[:,:,:3], torch.stack((torch.ones(N), torch.zeros(N), torch.zeros(N)), dim=1).to(projectionMatrix.device))
    v = torch.linalg.solve(mrt[:,:,:3], torch.stack((torch.zeros(N), torch.ones(N), torch.zeros(N)), dim=1).to(projectionMatrix.device))
    posTrans = lambda x: x[:,[2,0,1]]
    s = posTrans(s)
    d = posTrans(d)
    v = posTrans(v) * normV
    u = posTrans(u) * normU
    return torch.cat((s,d,v,u), dim=1)

class ProjectionGeom(torch.autograd.Function):
    normU = 1
    normV = 1
    @staticmethod
    def forward(ctx, input, projectionMatrix, label):
        geometry.parameters = getResultParams(projectionMatrix, ProjectionGeom.normU, ProjectionGeom.normV)
        result = ForwardProjection.apply(output.reshape(1, 1,beijingVolumeSize[2],beijingVolumeSize[1],beijingVolumeSize[0]))
        residual = label - input
        ctx.save_for_backward(projectionMatrix, result, residual)
        return torch.autograd.Variable(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        # Beta
        projectionMatrix, sino, residual = ctx.saved_tensors
        gMatrix = torch.zeros_like(projectionMatrix)
        gsinoX, = torch.gradient(sino, dim=3)
        gsinoY, = torch.gradient(sino, dim=4)
        gsinoX = gsinoX[...,::beijingPlanes,:,:].contiguous()
        gsinoY = gsinoY[...,::beijingPlanes,:,:].contiguous()
        projectionMatrix = projectionMatrix[...,::beijingPlanes,:,:]
        residual = residual[...,20:36,:,:].contiguous()
        gMatrix[...,::beijingPlanes,:,:] = dffg(gsinoX, gsinoY, residual, projectionMatrix.contiguous())
        # gMatrix = dffg(gsinoX, gsinoY, residual, projectionMatrix.contiguous())
        print(gMatrix[0,...])
        return None, gMatrix, None

# Beta
def constuctProjectionMatrix(angles, trans):
    return torch.stack([torch.cos(angles), -torch.sin(angles), torch.ones_like(angles) * trans[:,0],torch.ones_like(angles) * trans[:,0],
                        torch.sin(angles), torch.cos(angles), torch.ones_like(angles) * trans[:,1],torch.ones_like(angles) * trans[:,1],
                        torch.zeros_like(angles), torch.zeros_like(angles), torch.ones_like(angles),torch.ones_like(angles) * trans[:,1],
                        torch.zeros_like(angles), torch.zeros_like(angles), torch.ones_like(angles),torch.ones_like(angles) * trans[:,1]
                        ]).view(4,4,-1).permute(2,0,1).to(torch.float32).to(device)

def build_coordinate_test(L):
    input = torch.zeros(L, L, 2)
    value_xy = np.linspace(-1,1,L)
    for x in range(L):
        for y in range(L):
            input[x,y,0] = value_xy[x]
            input[x,y,1] = value_xy[y]
    return input.reshape(-1, 2).to(device)

if __name__ == '__main__':
    projectionPath = "/home/nv/wyk/Data/lz/projection.raw"
    outputPath = "/home/nv/wyk/Data/output.raw"
    projection = np.fromfile(projectionPath, dtype="float32")
    projection = torch.from_numpy(projection).reshape(1,1,1080*21,128,80).to(device)
    projection[torch.isnan(projection)] = 0
    projection = projection[...,2:78]
    bytelen = beijingVolumeSize[0]*beijingVolumeSize[1]
    input = build_coordinate_test(beijingVolumeSize[0])
    lossFunction = nn.MSELoss()
    optimizer = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in model]
    scheduler = [torch.optim.lr_scheduler.StepLR(o, step_size=50, gamma=0.95) for o in optimizer]
    projectionMatrixInit, normU, normV = getProjectionMatrix(geometry.parameters)
    projectionMatrixInit = projectionMatrixInit.to(device)
    ProjectionGeom.normU = normU.to(device)
    ProjectionGeom.normV = normV.to(device)
    anglesNum = beijingAngleNum * beijingPlanes
    projectionMatrixCorrAngle = torch.autograd.Variable(torch.tensor([0.0] * anglesNum).to(device), requires_grad=True)
    projectionMatrixCorrTrans = torch.autograd.Variable(torch.tensor([[0.0,0.0]] * anglesNum).to(device), requires_grad=True)
    optimizerCorr = torch.optim.Adam([projectionMatrixCorrAngle, projectionMatrixCorrTrans], lr=1e-2)
    tic = time.time()
    with tqdm.trange(1) as epochs:
        for e in epochs:
            output = torch.zeros(beijingVolumeSize[2], bytelen, 1).to(device)
            for s in range(beijingVolumeSize[2]):
                output[s,...] = model[s](input).float()
            output_projection = ForwardProjection.apply(output.reshape(1, 1,beijingVolumeSize[2],beijingVolumeSize[1],beijingVolumeSize[0]))
            loss = lossFunction(output_projection, projection)
            loss.backward()
            for s in range(beijingVolumeSize[2]):
                optimizer[s].step()
                optimizer[s].zero_grad()
                scheduler[s].step()
            if e%1==0:
                output.detach().cpu().numpy().astype("float32").tofile(outputPath)
            epochs.set_description(f"Learning projections ...")
            epochs.set_postfix({"loss": loss.item()})
        print("geometry finetune...")
        projectionMatrixCorr = constuctProjectionMatrix(projectionMatrixCorrAngle, projectionMatrixCorrTrans)
        projectionMatrixCorr = projectionMatrixCorr.reshape(anglesNum, 4, 4).to(device)
        projectionMatrix = projectionMatrixInit @ projectionMatrixCorr
        output_sino = ProjectionGeom.apply(output, projectionMatrix, torch.ones_like(output))
        loss = lossFunction(output_sino, torch.ones_like(output_sino))
        loss.backward()
        optimizerCorr.step()
        optimizerCorr.zero_grad()
    print("Training successfully. Now we are going to render...")
    output = torch.zeros(beijingVolumeSize[2], bytelen, 1).to(device)
    for s in range(beijingVolumeSize[2]):
        output[s,...] = model[s](input).float()
    output.detach().cpu().numpy().astype("float32").tofile(outputPath)
    print(f"Success. Render output is saved in {outputPath}. Total time cost:{time.time() - tic}s")