import torch
import torch.nn.functional as F
device = "cuda:0"

def translate(target, rotation, t):
    target = target.view(24, int(1080*128/24), 80*21)
    rotation = rotation.view(2, 2, 24).permute(2, 0, 1).contiguous()
    for angle in range(24):
        image = target[angle, ...]
        h, w = image.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        input_coords = torch.stack((grid_x, grid_y), dim=-1).float().view(-1,2).to(device)
        rot_matrix = rotation[angle, ...]
        result = torch.matmul(input_coords, rot_matrix).squeeze(1) + t[angle, ...]
        target[angle, ...] = F.grid_sample(image.unsqueeze(0).unsqueeze(0), result.reshape(1,h,w,2)).squeeze()
    return target

if __name__ == '__main__':
    theta = torch.autograd.Variable(torch.zeros(24, 1)).to(device)
    rotation_matrix = torch.stack([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)])
    translation_matrix = torch.autograd.Variable(torch.zeros(24, 2)).to(device)

    target = torch.ones(1,1,22680,128,80).to(device)
    result = translate(target, rotation_matrix, translation_matrix)