import cv2
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_ssim as ssim

def normalization(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.Laplacian(image, cv2.CV_32F)
    return image

def get_ssim(img1, img2, saveDiff=""):
    sino_raw = np.fromfile(img1, dtype="float32").reshape([1080,128,1680])
    sino_gen = np.fromfile(img2, dtype="float32").reshape([1080,128,1680])
    sino_raw = normalization(sino_raw[1010])
    sino_gen = normalization(sino_gen[1010])
    sino_raw = torch.from_numpy(sino_raw).reshape([1,1,128,1680]).cuda()
    sino_gen = torch.from_numpy(sino_gen).reshape([1,1,128,1680]).cuda()
    sino_raw = torch.autograd.Variable(sino_raw, requires_grad=False)
    sino_gen = torch.autograd.Variable(sino_gen, requires_grad=True)
    ssim_loss = 1 - ssim.ssim(sino_raw, sino_gen)
    loss = ssim_loss
    print(f"SSIM: {loss.item()}")
    loss.backward()
    if saveDiff != "":
        (sino_gen.grad.cpu().numpy()*10e5).astype("float32").tofile(saveDiff)

if __name__ == '__main__':
    get_ssim("/media/wyk/wyk/Data/raws/sino_raw.raw", "/media/wyk/wyk/Data/raws/sino_gen.raw", "/media/wyk/wyk/Data/raws/sino_p1.raw")
    get_ssim("/media/wyk/wyk/Data/raws/sino_mess.raw", "/media/wyk/wyk/Data/raws/sino_mess_gen.raw", "/media/wyk/wyk/Data/raws/sino_p2.raw")