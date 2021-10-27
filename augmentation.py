import random, math
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
from utils import rotate_bbox
import config as cfg

def Rotate(img, v, bboxes):  # [-45, 45]
    assert -45 <= v <= 45
    if random.random() > 0.5:
        v = -v
    img = img.rotate(v)
    if bboxes is not None:
        vertices = torch.stack([torch.stack([bboxes[:, 0], bboxes[:, 1]], dim=-1),
                            torch.stack([bboxes[:, 0], bboxes[:, 3]], dim=-1),
                            torch.stack([bboxes[:, 2], bboxes[:, 3]], dim=-1),
                            torch.stack([bboxes[:, 2], bboxes[:, 1]], dim=-1)], dim=1)
        radians = torch.as_tensor([math.radians(-v) for _ in range(vertices.size(0))]).float()
        anchors = (torch.as_tensor(cfg.size).float() / 2).unsqueeze(dim=0).repeat(vertices.size(0), 1)

        rotated_vertices = rotate_bbox(vertices, radians, anchors)
        n_bboxes = torch.stack([
            torch.clamp(torch.min(rotated_vertices[:, :, 0], dim=-1)[0], min=0, max=cfg.size[0]),
            torch.clamp(torch.min(rotated_vertices[:, :, 1], dim=-1)[0], min=0, max=cfg.size[1]),
            torch.clamp(torch.max(rotated_vertices[:, :, 0], dim=-1)[0], min=0, max=cfg.size[0]),
            torch.clamp(torch.max(rotated_vertices[:, :, 1], dim=-1)[0], min=0, max=cfg.size[1])], dim=-1)
        return img, n_bboxes
    else:
        return img, bboxes

def AutoContrast(img, _, bboxes):
    return PIL.ImageOps.autocontrast(img), bboxes

def Invert(img, _, bboxes):
    return PIL.ImageOps.invert(img), bboxes

def Equalize(img, _, bboxes):
    return PIL.ImageOps.equalize(img), bboxes

def HorizontalFlip(img, _, bboxes):
    if bboxes is not None:
        n_bboxes = bboxes.clone()
        n_bboxes[:, 0::2] = cfg.size[0] - n_bboxes[:, 0::2]
        n_bboxes = torch.stack([n_bboxes[:, 2], n_bboxes[:, 1], n_bboxes[:, 0], n_bboxes[:, 3]], dim=-1)
        return img.transpose(PIL.Image.FLIP_LEFT_RIGHT), n_bboxes
    else:
        return img.transpose(PIL.Image.FLIP_LEFT_RIGHT), bboxes

def VerticalFlip(img, _, bboxes):
    if bboxes is not None:
        n_bboxes = bboxes.clone()
        n_bboxes[:, 1::2] = cfg.size[1] - n_bboxes[:, 1::2]
        n_bboxes = torch.stack([n_bboxes[:, 0], n_bboxes[:, 3], n_bboxes[:, 2], n_bboxes[:, 1]], dim=-1)
        return img.transpose(PIL.Image.FLIP_TOP_BOTTOM), n_bboxes
    else:
        return img.transpose(PIL.Image.FLIP_TOP_BOTTOM), bboxes

def Solarize(img, v, bboxes):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v), bboxes

def SolarizeAdd(img, addition=0, bboxes=None, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold), bboxes

def Posterize(img, v, bboxes):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v), bboxes

def Contrast(img, v, bboxes):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v), bboxes

def Color(img, v, bboxes):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v), bboxes

def Brightness(img, v, bboxes):
    return PIL.ImageEnhance.Brightness(img).enhance(v), bboxes

def Sharpness(img, v, bboxes):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v), bboxes

def Cutout(img, v):
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v, bboxes):
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)

    if bboxes is not None:
        xy = torch.as_tensor(xy).unsqueeze(dim=0).repeat(bboxes.size(0), 1).long()
        dx = torch.clamp(
            torch.stack([xy[:, 2], bboxes[:, 2]], dim=-1).min(dim=-1)[0] -
            torch.stack([xy[:, 0], bboxes[:, 0]], dim=-1).max(dim=-1)[0],
            min=0)
        dy = torch.clamp(
            torch.stack([xy[:, 3], bboxes[:, 3]], dim=-1).min(dim=-1)[0] -
            torch.stack([xy[:, 1], bboxes[:, 1]], dim=-1).max(dim=-1)[0],
            min=0)
        intersection = dx * dy
        area = (bboxes[:, 2:] - bboxes[:, :2])[:, 0] * (bboxes[:, 2:] - bboxes[:, :2])[:, 1]
        conf = intersection / area
        valid_ind = torch.where(conf > 0.7, True, False)
        n_bboxes = bboxes[~valid_ind, :]
        return img, n_bboxes

    else:
        return img, bboxes

def source_augment():
    ops = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Posterize, 0, 20),
        (Solarize, 0, 24),
        (SolarizeAdd, 0, 64),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 3),
        (Sharpness, 0.1, 1.9),
        (CutoutAbs, 0, 2000),
        (HorizontalFlip, 0, 1),
        (VerticalFlip, 0, 1)
    ]
    return ops

def target_augment():
    ops = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Posterize, 0, 20),
        (Solarize, 0, 24),
        (SolarizeAdd, 0, 64),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 3),
        (Sharpness, 0.1, 1.9)
    ]
    return ops

class RandAugment:
    def __init__(self, n, m, augment_list):
        self._n = n
        self._m = m
        self._augment_list = augment_list

    def __call__(self, img, bboxes=None):
        ops = random.sample(self._augment_list, k=self._n)
        for op, minval, maxval in ops:
            val = (float(self._m) / 30) * float(maxval - minval) + minval
            img, bboxes = op(img, val, bboxes)
        return img, bboxes
