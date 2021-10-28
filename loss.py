import torch
import torchvision
from utils import Intergral
import config as cfg

class TargetCriterion(torch.nn.Module):
    def __init__(self):
        super(TargetCriterion, self).__init__()

    def _qfl_loss(self, pred_batch, target_batch, pos_mask_batch, total_mask_batch, beta=2):
        loss = []
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_batch, target_batch, reduction='none')
        bce_loss = bce_loss * (pred_batch.sigmoid() - target_batch).abs().pow(beta)
        bce_loss = bce_loss * total_mask_batch
        for pos_mask, l in zip(pos_mask_batch, bce_loss):
            loss.append(l.sum() / (pos_mask.sum() + 1e-3))
        return torch.stack(loss, dim=0).mean()

    def _dfl_loss(self, pred_batch, target_batch, pos_mask_batch):
        loss = []
        for pos_mask, pred, target in zip(pos_mask_batch, pred_batch, target_batch):
            target = target[:, pos_mask]
            pred = pred[:, :, pos_mask]

            target_left = target.long()
            target_right = target_left + 1
            weight_left = target_right.float() - target
            weight_right = target - target_left.float()

            ce_loss = torch.nn.functional.cross_entropy(pred, target_left, reduction='none') * weight_left + \
                    torch.nn.functional.cross_entropy(pred, target_right, reduction='none') * weight_right # (4, n_pos)
            ce_loss = ce_loss.sum() / (4 * pos_mask.sum() + 1e-3)
            loss.append(ce_loss)
        return torch.stack(loss, dim=0).mean()

    def forward(self, pseudo_qfl_batch, pseudo_dfl_batch, pred_cls_batch, pred_reg_batch, pos_mask_batch, total_mask_batch):
        qfl_loss = self._qfl_loss(pred_cls_batch, pseudo_qfl_batch, pos_mask_batch, total_mask_batch)
        dfl_loss = self._dfl_loss(pred_reg_batch, pseudo_dfl_batch, pos_mask_batch)
        return qfl_loss, dfl_loss

class SourceCriterion(torch.nn.Module):
    def __init__(self):
        super(SourceCriterion, self).__init__()
        self._intergral = Intergral()

    def _qfl_loss(self, pred_batch, target_batch, beta=2):
        pos_mask_batch = target_batch > 0
        loss = []
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_batch, target_batch, reduction='none')
        bce_loss = bce_loss * (pred_batch.sigmoid() - target_batch).abs().pow(beta)
        for pos_mask, l in zip(pos_mask_batch, bce_loss):
            loss.append(l.sum() / (pos_mask.sum() + 1e-3))
        return torch.stack(loss, dim=0).mean()

    def _dfl_loss(self, pred_batch, target_batch):
        pos_mask_batch = (~torch.isinf(target_batch)).sum(dim=1).bool()
        loss = []
        for pos_mask, pred, target in zip(pos_mask_batch, pred_batch, target_batch):
            target = target[:, pos_mask]
            pred = pred[:, :, pos_mask]

            target_left = target.long()
            target_right = target_left + 1
            weight_left = target_right.float() - target
            weight_right = target - target_left.float()

            ce_loss = torch.nn.functional.cross_entropy(pred, target_left, reduction='none') * weight_left + \
                    torch.nn.functional.cross_entropy(pred, target_right, reduction='none') * weight_right # (4, n_pos)
            ce_loss = ce_loss.sum() / (4 * pos_mask.sum() + 1e-3)
            loss.append(ce_loss)
        return torch.stack(loss, dim=0).mean()

    def _giou_loss(self, pred_batch, target_batch, anchor_points, fpn_strides):
        pos_mask_batch = (~torch.isinf(target_batch)).sum(dim=1).bool()
        pred_batch = pred_batch.softmax(dim=2)
        d_batch = self._intergral(pred_batch)
        loss = []
        for pos_mask, d, target in zip(pos_mask_batch, d_batch, target_batch):
            if pos_mask.sum() == 0:
                loss.append(torch.tensor(0., requires_grad=True).to(pred_batch.device))
            else:
                target = target[:, pos_mask].transpose(0, 1)
                d = d[:, pos_mask].transpose(0, 1)
                stride = fpn_strides[pos_mask]
                anchor_point = anchor_points[pos_mask, :] # (y, x) formula

                target = target * stride.unsqueeze(dim=1)
                d = d * stride.unsqueeze(dim=1)

                target_bboxes = torch.stack([
                    torch.clamp(anchor_point[:, 0] - target[:, 0], min=0, max=cfg.size[1]),
                    torch.clamp(anchor_point[:, 1] - target[:, 1], min=0, max=cfg.size[0]),
                    torch.clamp(anchor_point[:, 0] + target[:, 2], min=0, max=cfg.size[1]),
                    torch.clamp(anchor_point[:, 1] + target[:, 3], min=0, max=cfg.size[0]),
                ], dim=-1) # (y, x, y, x) formula
                pred_bboxes = torch.stack([
                    torch.clamp(anchor_point[:, 0] - d[:, 0], min=0, max=cfg.size[1]),
                    torch.clamp(anchor_point[:, 1] - d[:, 1], min=0, max=cfg.size[0]),
                    torch.clamp(anchor_point[:, 0] + d[:, 2], min=0, max=cfg.size[1]),
                    torch.clamp(anchor_point[:, 1] + d[:, 3], min=0, max=cfg.size[0])
                ], dim=-1) # (y, x, y, x) formula

                giou_loss = []
                for pred_bbox, target_bbox in zip(pred_bboxes, target_bboxes):
                    giou = torchvision.ops.generalized_box_iou(pred_bbox.unsqueeze(dim=0), target_bbox.unsqueeze(dim=0)).flatten()
                    giou_loss.append(1. - giou)
                giou_loss = torch.stack(giou_loss, dim=0).sum() / pos_mask.sum()
                
                loss.append(giou_loss)
        return torch.stack(loss, dim=0).mean()

    def forward(self, gt_qfl_batch, gt_dfl_batch, anchor_points, fpn_strides, pred_cls_batch, pred_reg_batch):
        qfl_loss = self._qfl_loss(pred_cls_batch, gt_qfl_batch)
        dfl_loss = self._dfl_loss(pred_reg_batch, gt_dfl_batch)
        giou_loss = self._giou_loss(pred_reg_batch, gt_dfl_batch, anchor_points, fpn_strides)

        return qfl_loss, dfl_loss, giou_loss
