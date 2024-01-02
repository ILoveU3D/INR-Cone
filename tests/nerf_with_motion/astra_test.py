import astra
import numpy as np
import torch

def _filter(projWidth):
    filter = np.ones([projWidth,1], dtype=np.float32)
    mid = np.floor(projWidth / 2)
    for i in range(projWidth):
        filter[i] = mid - np.abs(mid - i)
    return filter.T

# System matrix
anglesNum = 360
angles = np.linspace(0, 2*np.pi, anglesNum, endpoint=False)
# parameters = h5py.File("./projVec.mat", 'r')
detectorSize = 1080
volumeSize = [512, 512]
projectorGeometry = astra.create_proj_geom('fanflat', 1.0, detectorSize, angles, 800, 400)
volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1])
projector = astra.create_projector('cuda',projectorGeometry,volumeGeometry)
H = astra.OpTomo(projector)

# Reconstruction
lung = np.fromfile("/home/nv/wyk/Data/SheppLogan.raw","float32")
lung = lung.reshape(64,512,512)[32,...]
sino = H * lung.flatten()
sino = sino.reshape(1, anglesNum, detectorSize)
ramp = _filter(detectorSize)
sino = np.fft.ifft2(np.fft.fft2(sino) * ramp).real.flatten()
sino.tofile("/home/nv/wyk/Data/sino.raw")
recon = H.T * sino
recon /= (np.max(recon)+1e-4)
recon.tofile("/home/nv/wyk/Data/output.raw")