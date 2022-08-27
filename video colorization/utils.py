import torch
import torchvision.transforms as transforms
import numpy as np
from skimage import color


def unprocess(l, ab=None):
    if ab is not None:
        img_tensor = torch.cat([l, ab], dim=0)
        img_lab = np.array(img_tensor.permute(1, 2, 0).cpu())
        img_lab[:, :, 0] = (img_lab[:, :, 0] * 50.0) + 50.0
        img_lab[:, :, 0] = np.clip(img_lab[:, :, 0], 0, 100)
        img_lab[:, :, 1:] = img_lab[:, :, 1:] * 128.0
        img_lab[:, :, 1:] = np.clip(img_lab[:, :, 1:], -110, 110)
        img_rgb = color.lab2rgb(img_lab)
        img = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
    else:
        img_l = np.array(l.permute(1, 2, 0).cpu())
        img_l = ((img_l * 50.0) + 50.0) * (255 / 100)
        img = np.rint(img_l).astype(np.uint8)
    return img


def rgb_to_l(img_tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    img_tensor = inv_normalize(img_tensor)
    img = np.array(img_tensor.permute(1, 2, 0).cpu())
    img = color.rgb2gray(img)
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).float()
    img = torch.unsqueeze(img, 0).cuda()
    return img


def rgb2xyz(rgb):  # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > 0.04045).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (((rgb + 0.055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)

    x = (
        0.412453 * rgb[:, 0, :, :]
        + 0.357580 * rgb[:, 1, :, :]
        + 0.180423 * rgb[:, 2, :, :]
    )
    y = (
        0.212671 * rgb[:, 0, :, :]
        + 0.715160 * rgb[:, 1, :, :]
        + 0.072169 * rgb[:, 2, :, :]
    )
    z = (
        0.019334 * rgb[:, 0, :, :]
        + 0.119193 * rgb[:, 1, :, :]
        + 0.950227 * rgb[:, 2, :, :]
    )
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)
    return out


def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = (
        3.24048134 * xyz[:, 0, :, :]
        - 1.53715152 * xyz[:, 1, :, :]
        - 0.49853633 * xyz[:, 2, :, :]
    )
    g = (
        -0.96925495 * xyz[:, 0, :, :]
        + 1.87599 * xyz[:, 1, :, :]
        + 0.04155593 * xyz[:, 2, :, :]
    )
    b = (
        0.05564664 * xyz[:, 0, :, :]
        - 0.20404134 * xyz[:, 1, :, :]
        + 1.05731107 * xyz[:, 2, :, :]
    )

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))

    mask = (rgb > 0.0031308).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (1.055 * (rgb ** (1.0 / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1.0, 1.08883))[None, :, None, None]
    if xyz.is_cuda:
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > 0.008856).type(torch.FloatTensor)
    if xyz_scale.is_cuda:
        mask = mask.cuda()

    xyz_int = xyz_scale ** (1 / 3.0) * mask + (7.787 * xyz_scale + 16.0 / 116.0) * (
        1 - mask
    )

    L = 116.0 * xyz_int[:, 1, :, :] - 16.0
    a = 500.0 * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200.0 * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)
    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.0) / 116.0
    x_int = (lab[:, 1, :, :] / 500.0) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.0)
    if z_int.is_cuda:
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat(
        (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1
    )
    mask = (out > 0.2068966).type(torch.FloatTensor)
    if out.is_cuda:
        mask = mask.cuda()

    out = (out**3.0) * mask + (out - 16.0 / 116.0) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1.0, 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc
    return out


def rgb2lab(rgb, l_norm=1.0, l_cent=0, ab_norm=1.0):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0], :, :] - l_cent) / l_norm
    ab_rs = lab[:, 1:, :, :] / ab_norm
    out = torch.cat((l_rs, ab_rs), dim=1)
    return out


def lab2rgb(lab_rs, l_norm=1.0, l_cent=0, ab_norm=1.0):
    lum = lab_rs[:, [0], :, :] * l_norm + l_cent
    ab = lab_rs[:, 1:, :, :] * ab_norm
    lab = torch.cat((lum, ab), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out
