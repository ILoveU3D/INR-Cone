import torch
import os
os.chdir("/home/nv/wyk/inf-recon")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from nerf_cone_motion import getProjectionMatrix, getResultParams
from ConeBeamLayers.BeijingGeometry import parameters

device = "cuda:0"
params = parameters[:1,:].to(device)
print(params)
projectionMatrix, normU, normV = getProjectionMatrix(params)
print(projectionMatrix)
result = getResultParams(projectionMatrix, normU, normV)
print(result)
print(torch.nn.functional.mse_loss(params[:,:3], result[:,:3]))
print(torch.nn.functional.mse_loss(params[:,6:9], result[:,6:9]))