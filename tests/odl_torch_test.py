import odl
from odl.contrib import torch as odl_torch
import numpy as np
import torch

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

label = np.fromfile("/media/wyk/wyk/Data/raws/trainData/pa_1.raw", dtype="float32")
label = np.reshape(label, [64, 256, 256])
label = label[0, ...]
label /= np.max(label)
label = torch.from_numpy(label).cuda()
label = torch.autograd.Variable(label.reshape(1,256,256), requires_grad=True)
sino = Projection.apply(label)
sino += torch.empty_like(sino).normal_()
loss = torch.nn.functional.mse_loss(sino, torch.zeros_like(sino))
loss.backward()
label.grad.cpu().numpy().astype("float32").tofile("/media/wyk/wyk/Data/raws/output.raw")