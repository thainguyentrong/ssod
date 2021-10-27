import torch
import torchvision
from PIL import Image
import config as cfg
import math
from functools import partial
from sklearn.metrics import average_precision_score
import numpy as np

def unnormalize(image, mean, std):
    image = ((image * torch.as_tensor(std).reshape(1, image.size(1), 1, 1).to(image.device)) + torch.as_tensor(mean).reshape(1, image.size(1), 1, 1).to(image.device))
    return image

def disable_batchnorm_tracking(model):
    def fn(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
    model.apply(fn)

def enable_batchnorm_tracking(model):
    def fn(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = True
    model.apply(fn)

def load_categories():
    o2i = {obj: idx for idx, obj in enumerate(cfg.categories)}
    i2o = {idx: obj for idx, obj in enumerate(cfg.categories)}
    return o2i, i2o

def rotate_bbox(bboxes, radians, anchors):
    rotate_mat = torch.stack(
        [torch.cos(radians), -torch.sin(radians), torch.sin(radians), torch.cos(radians)],
    dim=-1)
    rotate_mat = rotate_mat.reshape(rotate_mat.size(0), 2, 2)
    anchors = anchors.unsqueeze(dim=1)
    rotated_bboxes = anchors + torch.matmul(rotate_mat, (bboxes - anchors).transpose(1, 2)).transpose(1, 2)
    return rotated_bboxes

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    results = map(pfunc, *args)
    return list(results)

def compute_loss_target_weight(n_iter, max_iter):
    mu = 0.5 - math.cos(min(math.pi, 2 * math.pi * n_iter / max_iter)) / 2
    return mu

class Inference:
    def __init__(self):
        self._gttrans = GtTransform()
        self._intergral = Intergral()

    def __call__(self, cls_dis, reg_dis, threshold):
        pred_quality_score_batch, pred_label_batch = cls_dis.max(dim=1)
        pred_quality_score_batch, pred_pos_ind_batch = torch.topk(pred_quality_score_batch, k=cfg.max_object_per_sample, dim=1)
        pred_d_batch = self._intergral(reg_dis)

        out_clses_batch, out_bboxes_batch, out_quality_scores_batch = [], [], []
        for pred_quality_score, pred_label, pred_d, pred_pos_ind in zip(pred_quality_score_batch, pred_label_batch, pred_d_batch, pred_pos_ind_batch):
            pred_label = pred_label[pred_pos_ind]
            
            k_ind = pred_quality_score >= threshold
            pred_quality_score = pred_quality_score[k_ind]
            pred_pos_ind = pred_pos_ind[k_ind]
            pseudo_label = pred_label[k_ind]

            pred_center = self._gttrans._anchor_points_with_scales.to(pred_pos_ind.device)[pred_pos_ind, :] # (y, x) formula
            pred_pos_d = pred_d[:, pred_pos_ind]
            pred_stride = self._gttrans._fpn_strides.to(pred_pos_ind.device)[pred_pos_ind]
            pred_pos_d = (pred_pos_d * pred_stride.unsqueeze(dim=0)).transpose(0, 1)

            pseudo_bbox = torch.stack([
                torch.clamp(pred_center[:, 0] - pred_pos_d[:, 0], min=0, max=cfg.size[1]),
                torch.clamp(pred_center[:, 1] - pred_pos_d[:, 1], min=0, max=cfg.size[0]),
                torch.clamp(pred_center[:, 0] + pred_pos_d[:, 2], min=0, max=cfg.size[1]),
                torch.clamp(pred_center[:, 1] + pred_pos_d[:, 3], min=0, max=cfg.size[0])
            ], dim=-1) # (y, x) formula

            mask = pseudo_label != 0 # positive isn't no_object label
            pseudo_bbox = pseudo_bbox[mask, :]
            pseudo_label = pseudo_label[mask]
            pred_quality_score = pred_quality_score[mask]

            iou_ind = torchvision.ops.nms(pseudo_bbox, pred_quality_score, iou_threshold=0.7)
            out_bboxes_batch.append(pseudo_bbox[iou_ind, :])
            out_clses_batch.append(pseudo_label[iou_ind])
            out_quality_scores_batch.append(pred_quality_score[iou_ind])

        return out_clses_batch, out_bboxes_batch, out_quality_scores_batch

class Evaluation:
    def __init__(self):
        self._gt_clses, self._gt_bboxes, self._pred_clses, self._pred_bboxes, self._pred_quality_scores = [], [], [], [], []
        self._threshold = 0.7
    
    def append(self, gt_clses_batch, gt_bboxes_batch, pred_clses_batch, pred_bboxes_batch, pred_quality_scores_batch):
        self._gt_clses += gt_clses_batch
        self._gt_bboxes += gt_bboxes_batch
        self._pred_clses += pred_clses_batch
        self._pred_bboxes += pred_bboxes_batch
        self._pred_quality_scores += pred_quality_scores_batch

    def eval(self):
        Y_true, Y_score = [], []
        for (gt_clses, gt_bboxes, pred_clses, pred_bboxes, pred_quality_scores) in zip(self._gt_clses, self._gt_bboxes, self._pred_clses, self._pred_bboxes, self._pred_quality_scores):
            for (pred_cls, pred_bbox, pred_quality_score) in zip(pred_clses, pred_bboxes, pred_quality_scores):
                if gt_bboxes.size(0) > 0: 
                    overlap = torchvision.ops.box_iou(pred_bbox.unsqueeze(dim=0), gt_bboxes).squeeze(dim=0)
                    max_iou, max_ind = overlap.max(dim=0)
                    if max_iou > self._threshold and pred_cls == gt_clses[max_ind]:
                        Y_true.append(1)
                        gt_bboxes = torch.cat([gt_bboxes[:max_ind, :], gt_bboxes[max_ind + 1 :, :]])
                        gt_clses = torch.cat([gt_clses[:max_ind], gt_clses[max_ind + 1 :]])
                    else:
                        Y_true.append(0)
                    Y_score.append(pred_quality_score.item())

                else:
                    Y_true.append(0)
                    Y_score.append(pred_quality_score.item())
        if len(Y_true) == 0:
            return 0.
        return average_precision_score(np.array(Y_true), np.array(Y_score))


class Resize:
    def __init__(self, nsize):
        self._nsize = nsize
        
    def __call__(self, image, bboxes=None):
        old_size = (image.width, image.height)
        factor_x = image.width / self._nsize[0]
        factor_y = image.height / self._nsize[1]
        factor = max(factor_x, factor_y)
        new_size = (min(self._nsize[0], int(image.width / factor)), min(self._nsize[1], int(image.height / factor)))
        image = image.resize(size=new_size)
        new_image = Image.new('RGB', self._nsize, color=0)
        new_image.paste(image, ((self._nsize[0] - new_size[0]) // 2, (self._nsize[1] - new_size[1]) // 2))

        if bboxes is None:
            return new_image
        else:    
            bboxes *= (torch.as_tensor(new_size) / torch.as_tensor(old_size)).repeat(2).unsqueeze(dim=0)
            bboxes += torch.as_tensor(((self._nsize[0] - new_size[0]) // 2, (self._nsize[1] - new_size[1]) // 2)).repeat(2).unsqueeze(dim=0)
            return new_image, bboxes

class Intergral(torch.nn.Module):
    def __init__(self):
        super(Intergral, self).__init__()
        self._range = torch.arange(0, cfg.reg_max + 1).reshape(1, 1, -1, 1)

    def forward(self, x):
        # input: (b, 4, reg_max + 1, h * w)
        x = (x * self._range.to(x.device)).sum(dim=2)
        return x

class Upsample(torch.nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, fsize):
        x = torch.nn.functional.interpolate(x, size=fsize, mode='bilinear', align_corners=True)
        return x

class ResidualBlock(torch.nn.Module):
    _expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self._out_channels = out_channels
        self._conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self._bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self._conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        self._shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != self._expansion * out_channels:
            self._shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=self._expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(num_features=self._expansion * out_channels)
            )

    def forward(self, x):
        out = self._conv1(x)
        out = torch.nn.functional.relu(self._bn1(out))  

        out = self._conv2(out)
        out = self._bn2(out)

        out += self._shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

class GtTransform(torch.nn.Module):
    def __init__(self):
        super(GtTransform, self).__init__()
        self._anchor_points_with_scales, self._anchor_boxes, self._fpn_strides = [], [], []
        fscales = []
        for fpn_stride in cfg.fpn_strides:
            fscales.append((cfg.size[1] // fpn_stride, cfg.size[0] // fpn_stride))
        for fscale, fpn_stride in zip(fscales, cfg.fpn_strides):
            y, x = torch.meshgrid(
                torch.arange(0, fscale[0]),
                torch.arange(0, fscale[1])
            )
            self._anchor_points_with_scales.append(torch.stack([y.flatten() + 0.5, x.flatten() + 0.5], dim=-1) * fpn_stride)
            self._anchor_boxes.append(torch.stack([
                y.flatten() * fpn_stride,
                x.flatten() * fpn_stride,
                (y.flatten() + 1) * fpn_stride,
                (x.flatten() + 1) * fpn_stride], dim=-1))
            self._fpn_strides.append(torch.empty(fscale[0] * fscale[1]).fill_(fpn_stride))

        self._anchor_points_with_scales = torch.cat(self._anchor_points_with_scales, dim=0).float() # (n_flatten, 2)
        self._anchor_boxes = torch.cat(self._anchor_boxes, dim=0).float() # (n_flatten, 4)
        self._fpn_strides = torch.cat(self._fpn_strides, dim=0).float() # (n_flatten)
        self._intergral = Intergral()

    def _single_process(self, gt_clses, gt_bboxes, pred_d):
        gt_qfl = torch.empty((len(cfg.categories), self._anchor_points_with_scales.size(0))).fill_(0).float().to(gt_bboxes.device) # (n_categories, n_flatten)
        gt_dfl = torch.empty((self._anchor_boxes.size(1), self._anchor_boxes.size(0))).fill_(float('inf')).to(gt_bboxes.device) # (4, n_flatten)

        self._anchor_points_with_scales = self._anchor_points_with_scales.to(gt_bboxes.device)
        self._anchor_boxes = self._anchor_boxes.to(gt_bboxes.device)
        self._fpn_strides = self._fpn_strides.to(gt_bboxes.device)

        gt_cen_bboxes = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2 # (y, x) formula
        g2a_dist = torch.cdist(gt_cen_bboxes, self._anchor_points_with_scales, p=2.0)
        _, k_inds = torch.topk(g2a_dist, k=45, dim=1, largest=False)

        for gt_cls, gt_bbox, k_ind in zip(gt_clses, gt_bboxes, k_inds):
            k_anchors = self._anchor_boxes[k_ind, :] # (k, 4)
            Dg = torchvision.ops.box_iou(gt_bbox.unsqueeze(dim=0), k_anchors).squeeze(dim=0) # (k)
            tg = Dg.mean() + Dg.std()
            pos_ind = k_ind[Dg >= tg]

            for i in pos_ind:
                anchor_point = self._anchor_points_with_scales[i, :] # (y, x) formula
                fpn_stride = self._fpn_strides[i]

                if (gt_bbox[0] <= anchor_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= anchor_point[1] <= gt_bbox[3]):
                    dtop = (anchor_point[0] - gt_bbox[0]) / fpn_stride
                    dbottom = (gt_bbox[2] - anchor_point[0]) / fpn_stride
                    dleft = (anchor_point[1] - gt_bbox[1]) / fpn_stride
                    dright = (gt_bbox[3] - anchor_point[1]) / fpn_stride
                    gt_d = torch.stack([dtop, dleft, dbottom, dright], dim=0)

                    quality_score = self._overlap_tlbr(gt_d.unsqueeze(dim=0), pred_d[:, i].unsqueeze(dim=0), anchor_point.unsqueeze(dim=0), fpn_stride.unsqueeze(dim=0))

                    gt_qfl[gt_cls, i] = quality_score.squeeze(dim=0)
                    gt_dfl[:, i] = gt_d

        return gt_qfl, gt_dfl
    
    def _overlap_tlbr(self, d_1, d_2, anchor_point, stride):
        d_1 = d_1 * stride
        d_2 = d_2 * stride
        bbox_1 = torch.stack([
            torch.clamp(anchor_point[:, 0] - d_1[:, 0], min=0, max=cfg.size[1]),
            torch.clamp(anchor_point[:, 1] - d_1[:, 1], min=0, max=cfg.size[0]),
            torch.clamp(anchor_point[:, 0] + d_1[:, 2], min=0, max=cfg.size[1]),
            torch.clamp(anchor_point[:, 1] + d_1[:, 3], min=0, max=cfg.size[0])], dim=1) # (y, x, y, x) formula
        bbox_2 = torch.stack([
            torch.clamp(anchor_point[:, 0] - d_2[:, 0], min=0, max=cfg.size[1]),
            torch.clamp(anchor_point[:, 1] - d_2[:, 1], min=0, max=cfg.size[0]),
            torch.clamp(anchor_point[:, 0] + d_2[:, 2], min=0, max=cfg.size[1]),
            torch.clamp(anchor_point[:, 1] + d_2[:, 3], min=0, max=cfg.size[0])], dim=1) # (y, x, y, x) formula

        return torchvision.ops.box_iou(bbox_1, bbox_2)

    def __call__(self, gt_clses_batch, gt_bboxes_batch, pred_reg_batch):
        pred_reg_batch = pred_reg_batch.softmax(dim=2) # (b, 4, reg_max + 1, n_flatten)
        pred_d_batch = [pred_d for pred_d in self._intergral(pred_reg_batch)]

        results = multi_apply(self._single_process, gt_clses_batch, gt_bboxes_batch, pred_d_batch)
        gt_qfl_batch, gt_dfl_batch = list(zip(*results))
        gt_qfl_batch = torch.stack(gt_qfl_batch, dim=0)
        gt_dfl_batch = torch.stack(gt_dfl_batch, dim=0)

        return gt_qfl_batch, gt_dfl_batch, self._anchor_points_with_scales, self._fpn_strides
