import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from utils import load_categories, Resize
import config as cfg
from augmentation import RandAugment, source_augment, target_augment

c2i, i2c = load_categories()

class PictorV3Dataset(Dataset):
    def __init__(self, image_dir, label_dir, set_name, scale, transform):
        super(PictorV3Dataset, self).__init__()
        self._set_name = set_name
        self._transform = transform

        self._image_paths = []
        self._labels = dict()

        if label_dir is not None:
            for label_fname in os.listdir(label_dir):
                self._image_paths.append(image_dir + label_fname[:-4] + '.jpg')
                with open(label_dir + label_fname, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        category, xmin, ymin, xmax, ymax = line.strip().split(' ')
                        if image_dir + label_fname[:-4] + '.jpg' not in self._labels:
                            self._labels[image_dir + label_fname[:-4] + '.jpg'] = [[c2i[category], float(xmin), float(ymin), float(xmax), float(ymax)]]
                        else:
                            self._labels[image_dir + label_fname[:-4] + '.jpg'] += [[c2i[category], float(xmin), float(ymin), float(xmax), float(ymax)]]
        else:
            for fname in os.listdir(image_dir):
                self._image_paths.append(image_dir + fname)

        if scale is 'small':
            self._image_paths = self._image_paths[:100]
        elif scale is 'large':
            self._image_paths = self._image_paths
    
    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index]).convert('RGB')

        if self._set_name == 'source':
            weak_source, strong_source, gt_clses, gt_bboxes, trans_gt_bboxes = self._transform(image, self._labels[self._image_paths[index]], set_name=self._set_name)
            return weak_source, strong_source, gt_clses, gt_bboxes, trans_gt_bboxes

        elif self._set_name == 'target':
            weak_target, strong_target = self._transform(image, set_name=self._set_name)
            return weak_target, strong_target

        elif self._set_name == 'valid':
            image, gt_clses, gt_bboxes = self._transform(image, self._labels[self._image_paths[index]], set_name=self._set_name)
            return image, gt_clses, gt_bboxes

class Transformer:
    def __init__(self):
        self._resize = Resize(cfg.size)
        self._weaK_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        self._strong_source_aug = RandAugment(3, 5, source_augment())
        self._strong_target_aug = RandAugment(3, 5, target_augment())
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std)
        ])
    
    def __call__(self, image, label=None, set_name=None):
        if label:
            gt_clses, gt_bboxes = torch.as_tensor(label)[:, 0].long(), torch.as_tensor(label)[:, 1:].float()
            image, gt_bboxes = self._resize(image, gt_bboxes)
        else:
            image = self._resize(image)

        if set_name == 'source':
            weak_source = self._weaK_aug(image)
            strong_source, trans_gt_bboxes = self._strong_source_aug(image, gt_bboxes)
            weak_source = self._transform(weak_source)
            strong_source = self._transform(strong_source)
            gt_bboxes = torch.stack([gt_bboxes[:, 1], gt_bboxes[:, 0], gt_bboxes[:, 3], gt_bboxes[:, 2]], dim=-1) # convert (x, y, x, y) to (y, x, y, x) formula
            trans_gt_bboxes = torch.stack([trans_gt_bboxes[:, 1], trans_gt_bboxes[:, 0], trans_gt_bboxes[:, 3], trans_gt_bboxes[:, 2]], dim=-1) # convert (x, y, x, y) to (y, x, y, x) formula
            return weak_source, strong_source, gt_clses, gt_bboxes, trans_gt_bboxes

        elif set_name == 'target':
            weak_target = self._weaK_aug(image)
            strong_target, _ = self._strong_target_aug(image, None)
            weak_target = self._transform(weak_target)
            strong_target = self._transform(strong_target)
            return weak_target, strong_target
        
        elif set_name == 'valid':
            image = self._transform(image)
            gt_bboxes = torch.stack([gt_bboxes[:, 1], gt_bboxes[:, 0], gt_bboxes[:, 3], gt_bboxes[:, 2]], dim=-1) # convert (x, y, x, y) to (y, x, y, x) formula
            return image, gt_clses, gt_bboxes

def collate_source_fn(batch):
    weak_source = torch.stack([sample[0] for sample in batch], dim=0)
    strong_source = torch.stack([sample[1] for sample in batch], dim=0)
    gt_clses = [sample[2] for sample in batch]
    gt_bboxes = [sample[3] for sample in batch]
    trans_gt_bboxes = [sample[4] for sample in batch]
    return weak_source, strong_source, gt_clses, gt_bboxes, trans_gt_bboxes

def collate_target_fn(batch):
    weak_target = torch.stack([sample[0] for sample in batch], dim=0)
    strong_target = torch.stack([sample[1] for sample in batch], dim=0)
    return weak_target, strong_target

def collate_valid_fn(batch):
    image = torch.stack([sample[0] for sample in batch], dim=0)
    gt_clses = [sample[1] for sample in batch]
    gt_bboxes = [sample[2] for sample in batch]
    return image, gt_clses, gt_bboxes











# if __name__ == '__main__':
#     import cv2, tqdm
#     from utils import unnormalize, GtTransform
#     from torch.utils.data import DataLoader
#     from model import Model

#     dataset = PictorV3Dataset(image_dir='./dataset/source/images/', label_dir='./dataset/source/labels/', set_name='source', scale='large', transform=Transformer())
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, collate_fn=collate_source_fn, pin_memory=False)
#     gttrans = GtTransform()
#     model = Model()

#     for weak_image, strong_image, gt_clses_batch, gt_bboxes_batch, trans_gt_bboxes_batch in tqdm.tqdm(dataloader):        
#         # weak_pred_cls_batch, weak_pred_reg_batch = model(weak_image)
#         # strong_pred_cls_batch, strong_pred_reg_batch = model(strong_image)
#         weak_pred_reg_batch = torch.rand(1, 4, cfg.reg_max + 1, 21844)
#         strong_pred_reg_batch = torch.rand(1, 4, cfg.reg_max + 1, 21844)
    
#         weak_image = unnormalize(weak_image, mean=cfg.mean, std=cfg.std)
#         weak_image = (weak_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8').copy()
#         weak_image = cv2.cvtColor(weak_image, cv2.COLOR_BGR2RGB)

#         gt_qfl_batch, gt_dfl_batch, anchor_center_grid, strides = gttrans(gt_clses_batch, gt_bboxes_batch, weak_pred_reg_batch)

#         weak_label = (gt_qfl_batch[0] > 0).nonzero()[:, 0]
#         weak_pos_inds = (gt_qfl_batch[0] > 0).nonzero()[:, 1]
#         mask = weak_label > 0
#         weak_label = weak_label[mask]
#         weak_pos_inds = weak_pos_inds[mask]

#         weak_centers = torch.index_select(anchor_center_grid, dim=0, index=weak_pos_inds)
#         weak_d = torch.index_select(gt_dfl_batch[0], dim=1, index=weak_pos_inds)
#         weak_stride = torch.index_select(strides, dim=0, index=weak_pos_inds)
#         weak_d = (weak_d * weak_stride.unsqueeze(dim=0)).transpose(0, 1)

#         for ((cy, cx), (dtop, dleft, dbottom, dright), wlabel, stride) in zip(weak_centers, weak_d, weak_label.cpu().numpy().astype('int'), weak_stride):
#             xmin = (cx - dleft).cpu().numpy().astype('int')
#             xmax = (cx + dright).cpu().numpy().astype('int')
#             ymin = (cy - dtop).cpu().numpy().astype('int')
#             ymax = (cy + dbottom).cpu().numpy().astype('int')
#             cv2.rectangle(weak_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(weak_image, i2c[wlabel] + ' %d' % (stride), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
#             cv2.circle(weak_image, (cx, cy), 2, (0, 255, 0), -1)

#         for (label, (ymin, xmin, ymax, xmax)) in zip(gt_clses_batch[0].cpu().numpy().astype('int'), gt_bboxes_batch[0].cpu().numpy().astype('int')):
#             cv2.rectangle(weak_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
#             cv2.putText(weak_image, i2c[label], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)





#         strong_image = unnormalize(strong_image, mean=cfg.mean, std=cfg.std)
#         strong_image = (strong_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8').copy()
#         strong_image = cv2.cvtColor(strong_image, cv2.COLOR_BGR2RGB)

#         gt_qfl_batch, gt_dfl_batch, anchor_center_grid, strides = gttrans(gt_clses_batch, trans_gt_bboxes_batch, strong_pred_reg_batch)

#         strong_label = (gt_qfl_batch[0] > 0).nonzero()[:, 0]
#         strong_pos_inds = (gt_qfl_batch[0] > 0).nonzero()[:, 1]
#         mask = strong_label > 0
#         strong_label = strong_label[mask]
#         strong_pos_inds = strong_pos_inds[mask]

#         strong_centers = torch.index_select(anchor_center_grid, dim=0, index=strong_pos_inds)
#         strong_d = torch.index_select(gt_dfl_batch[0], dim=1, index=strong_pos_inds)
#         strong_stride = torch.index_select(strides, dim=0, index=strong_pos_inds)
#         strong_d = (strong_d * strong_stride.unsqueeze(dim=0)).transpose(0, 1)

#         for ((cy, cx), (dtop, dleft, dbottom, dright), slabel, stride) in zip(strong_centers, strong_d, strong_label.cpu().numpy().astype('int'), strong_stride):
#             xmin = (cx - dleft).cpu().numpy().astype('int')
#             xmax = (cx + dright).cpu().numpy().astype('int')
#             ymin = (cy - dtop).cpu().numpy().astype('int')
#             ymax = (cy + dbottom).cpu().numpy().astype('int')
#             cv2.rectangle(strong_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(strong_image, i2c[slabel] + ' %d' % (stride), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
#             cv2.circle(strong_image, (cx, cy), 2, (0, 255, 0), -1)

#         for (label, (ymin, xmin, ymax, xmax)) in zip(gt_clses_batch[0].cpu().numpy().astype('int'), trans_gt_bboxes_batch[0].cpu().numpy().astype('int')):
#             cv2.rectangle(strong_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
#             cv2.putText(strong_image, i2c[label], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)


#         cv2.imshow('weak_image', weak_image)
#         cv2.imshow('strong_image', strong_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
