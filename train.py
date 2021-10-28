import torch
import os, time, sys, cv2, tqdm
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import PictorV3Dataset, Transformer, collate_source_fn, collate_target_fn, collate_valid_fn, i2c
import config as cfg
from model import Model
from loss import SourceCriterion, TargetCriterion
from utils import unnormalize, disable_batchnorm_tracking, enable_batchnorm_tracking, Inference, compute_loss_target_weight, Intergral, GtTransform, Evaluation

use_gpu = torch.cuda.is_available()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, torch.nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, torch.nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, torch.nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, torch.nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def visualize(image, fname):
    fig, axs = plt.subplots()

    axs.imshow(image)
    axs.axis('off')
    fig.tight_layout()
    # plt.show()
    plt.savefig(fname + '.svg', format='svg', dpi=1000)
    plt.close('all')

def train():
    source_train_dataset = PictorV3Dataset(image_dir='./dataset/source/images/', label_dir='./dataset/source/labels/', set_name='source', scale='large', transform=Transformer())
    target_train_dataset = PictorV3Dataset(image_dir='./dataset/target/images/', label_dir=None, set_name='target', scale='large', transform=Transformer())
    valid_dataset = PictorV3Dataset(image_dir='./dataset/valid/images/', label_dir='./dataset/valid/labels/', set_name='valid', scale='large', transform=Transformer())

    source_train_loader = DataLoader(source_train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.n_jobs, collate_fn=collate_source_fn, pin_memory=use_gpu)
    target_train_loader = DataLoader(target_train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.n_jobs, collate_fn=collate_target_fn, pin_memory=use_gpu)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=cfg.n_jobs, collate_fn=collate_valid_fn, pin_memory=use_gpu)

    model = torch.nn.DataParallel(Model())
    src_criterion = SourceCriterion()
    tgt_criterion = TargetCriterion()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.lr)

    inference = Inference()
    intergral = Intergral()
    gttrans = GtTransform()
    eval = Evaluation()

    if use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        src_criterion.cuda()
        tgt_criterion.cuda()

    print('Number of parameters: ', count_parameters(model))

    curr_epoch = 0
    step_counter = 0
    n_batches = min(len(source_train_loader), len(target_train_loader))
    if os.path.exists(cfg.logdir + 'training_state.pth'):
        if use_gpu:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(cfg.logdir + 'training_state.pth', map_location=map_location)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        curr_epoch = ckpt['global_step']
        step_counter = ckpt['local_step']
        print('Restore model')

    
    for epoch in range(curr_epoch+1, cfg.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, ((weak_source, strong_source, gt_clses_batch, gt_bboxes_batch, trans_gt_bboxes_batch),
                    (weak_target, strong_target)) in enumerate(zip(source_train_loader, target_train_loader)):
            start = time.time()
            step_counter += 1

            if use_gpu:
                weak_source, strong_source = weak_source.cuda(), strong_source.cuda()
                weak_target, strong_target = weak_target.cuda(), strong_target.cuda()
                gt_clses_batch = [gt_clses.cuda() for gt_clses in gt_clses_batch]
                gt_bboxes_batch = [gt_bboxes.cuda() for gt_bboxes in gt_bboxes_batch]
                trans_gt_bboxes_batch = [trans_gt_bboxes.cuda() for trans_gt_bboxes in trans_gt_bboxes_batch]

            combined_total = torch.cat([weak_source, strong_source, weak_target, strong_target], dim=0)
            combined_source = torch.cat([weak_source, strong_source], dim=0)
            n_sources = combined_source.size(0)

            total_cls_logit_batch, total_reg_logit_batch = model(combined_total)
            source_cls_logit_batch_p, source_reg_logit_batch_p = total_cls_logit_batch[: n_sources], total_reg_logit_batch[: n_sources]

            disable_batchnorm_tracking(model) # doesn't update certain layers
            source_cls_logit_batch_pp, source_reg_logit_batch_pp = model(combined_source)
            enable_batchnorm_tracking(model)

            ## perform random logit interpolation
            cls_lambd = torch.rand(n_sources, source_cls_logit_batch_p.size(1), 1).to(source_cls_logit_batch_p.device)
            final_cls_logit_batch = (cls_lambd * source_cls_logit_batch_p) + ((1 - cls_lambd) * source_cls_logit_batch_pp) # (n_sources, n_classes, -1)

            reg_lambd = torch.rand(n_sources, 1, source_reg_logit_batch_p.size(2), 1).to(source_reg_logit_batch_p.device)
            final_reg_logit_batch = (reg_lambd * source_reg_logit_batch_p) + ((1 - reg_lambd) * source_reg_logit_batch_pp) # (n_sources, 4, reg_max + 1, -1)

            ## softmax for logits of weakly augmented source images
            weak_source_cls_logit_batch = final_cls_logit_batch[: weak_source.size(0)]
            strong_source_cls_logit_batch = final_cls_logit_batch[weak_source.size(0):]
            weak_source_cls_distribution = weak_source_cls_logit_batch.sigmoid()

            weak_source_reg_logit_batch = final_reg_logit_batch[: weak_source.size(0)]
            strong_source_reg_logit_batch = final_reg_logit_batch[weak_source.size(0):]

            ## softmax for logits of weakly augmented target images
            target_cls_logit_batch = total_cls_logit_batch[n_sources:]
            weak_target_cls_logit_batch = target_cls_logit_batch[: weak_target.size(0)]
            strong_target_cls_logit_batch = target_cls_logit_batch[weak_target.size(0):]
            final_cls_distribution = weak_target_cls_logit_batch.sigmoid()

            target_reg_logit_batch = total_reg_logit_batch[n_sources:]
            weak_target_reg_logit_batch = target_reg_logit_batch[: weak_target.size(0)]
            strong_target_reg_logit_batch = target_reg_logit_batch[weak_target.size(0):]
            final_reg_distribution = weak_target_reg_logit_batch.softmax(dim=2)

            ## perform relative confidence thresholding
            weak_gt_qfl_batch, weak_gt_dfl_batch, anchor_points, fpn_strides = gttrans(gt_clses_batch, gt_bboxes_batch, weak_source_reg_logit_batch.detach())
            strong_gt_qfl_batch, strong_gt_dfl_batch, _, _ = gttrans(gt_clses_batch, trans_gt_bboxes_batch, strong_source_reg_logit_batch.detach())

            pos_threshold, neg_threshold = [], []
            for pred, gt in zip(weak_source_cls_distribution.detach().reshape(cfg.batch_size, -1), weak_gt_qfl_batch.reshape(cfg.batch_size, -1)):
                pos_threshold.append(torch.index_select(input=pred, dim=0, index=(gt > 0).nonzero(as_tuple=False)[:, 0]))
                neg_threshold.append(torch.index_select(input=pred, dim=0, index=(gt == 0).nonzero(as_tuple=False)[:, 0]))

            pos_threshold = cfg.tau * torch.cat(pos_threshold, dim=0).mean()
            neg_threshold = torch.cat(neg_threshold, dim=0).mean()

            ## perform qfl and dfl pseudo label and mask from weak target
            final_max_scores, final_max_id_cls = final_cls_distribution.detach().max(dim=1) # (b, n_flatten), (b, n_flatten)
            final_pos_mask_batch = (final_max_scores >= pos_threshold) & (final_max_id_cls > 0)
            final_neg_mask_batch = final_max_scores <= neg_threshold

            final_total_mask = final_pos_mask_batch | final_neg_mask_batch
            pseudo_qfl_batch = final_cls_distribution.detach()
            pseudo_dfl_batch = intergral(final_reg_distribution.detach()) # (b, 4, n_flatten)

            ## perform losses
            weak_source_qfl_loss, weak_source_dfl_loss, weak_source_giou_loss = src_criterion(weak_gt_qfl_batch, weak_gt_dfl_batch, anchor_points, fpn_strides, weak_source_cls_logit_batch, weak_source_reg_logit_batch)
            strong_source_qfl_loss, strong_source_dfl_loss, strong_source_giou_loss = src_criterion(strong_gt_qfl_batch, strong_gt_dfl_batch, anchor_points, fpn_strides, strong_source_cls_logit_batch, strong_source_reg_logit_batch)
            strong_target_qfl_loss, strong_target_dfl_loss = tgt_criterion(pseudo_qfl_batch, pseudo_dfl_batch, strong_target_cls_logit_batch, strong_target_reg_logit_batch, final_pos_mask_batch, final_total_mask.unsqueeze(dim=1))

            weak_source_loss = weak_source_qfl_loss + weak_source_dfl_loss + 2 * weak_source_giou_loss
            strong_source_loss = strong_source_qfl_loss + strong_source_dfl_loss + 2 * strong_source_giou_loss
            strong_target_loss = strong_target_qfl_loss + strong_target_dfl_loss

            mu = compute_loss_target_weight(step_counter, cfg.epochs * n_batches)
            loss = weak_source_loss + strong_source_loss + mu * strong_target_loss

            loss.backward()

            if (step+1) % cfg.update == 0 or (step+1) == len(source_train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            end = time.time()
            sys.stdout.write('\rEpoch: %03d, Step: %04d/%d, Mu: %.6f, Obj/No Obj: %.5f/%.5f, Loss: %.9f, Time training: %.2f secs' % (epoch, step+1, n_batches, mu, pos_threshold.item(), neg_threshold.item(), loss.item(), end-start))


        if epoch % cfg.save == 0 or epoch == cfg.epochs:
            ckpt_path = cfg.logdir + 'training_state.pth'
            torch.save({'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': epoch,
                        'local_step': step_counter}, ckpt_path)

            ckpt_path = cfg.logdir + 'parameters.pth'
            torch.save({'state_dict': model.state_dict()}, ckpt_path)



        if epoch % cfg.evaluate == 0 or epoch == cfg.epochs:
            with torch.no_grad():
                pseudo_clses_batch, pseudo_bboxes_batch, pseudo_quality_scores_batch = inference(pseudo_qfl_batch, pseudo_dfl_batch, pos_threshold)
                weak_target = unnormalize(weak_target, mean=cfg.mean, std=cfg.std)
                weak_target = (weak_target[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8').copy()

                for (cls_id, (ymin, xmin, ymax, xmax), score) in zip(pseudo_clses_batch[0].cpu().numpy().astype('int'), pseudo_bboxes_batch[0].cpu().numpy().astype('int'), pseudo_quality_scores_batch[0].cpu().numpy()):
                    cv2.rectangle(weak_target, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(weak_target, i2c[cls_id] + ' %.2f' % (score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                visualize(weak_target, 'pseudo')

            model.eval()
            for image, gt_clses_batch, gt_bboxes_batch in tqdm.tqdm(valid_loader):
                if use_gpu:
                    image = image.cuda()
                    gt_clses_batch = [gt_clses.cuda() for gt_clses in gt_clses_batch]
                    gt_bboxes_batch = [gt_bboxes.cuda() for gt_bboxes in gt_bboxes_batch]

                with torch.no_grad():
                    pred_cls_batch, pred_reg_batch = model(image)
                    pred_cls_batch = pred_cls_batch.sigmoid()
                    pred_reg_batch = pred_reg_batch.softmax(dim=2)
                    pred_d_batch = intergral(pred_reg_batch)

                    pred_clses_batch, pred_bboxes_batch, pred_quality_scores_batch = inference(pred_cls_batch, pred_d_batch, threshold=0.5)

                    eval.append(gt_clses_batch, gt_bboxes_batch, pred_clses_batch, pred_bboxes_batch, pred_quality_scores_batch)

            mAP = eval.eval()
            print('mAP:', mAP)

            image = unnormalize(image, mean=cfg.mean, std=cfg.std)
            image = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8').copy()

            for (cls_id, (ymin, xmin, ymax, xmax), score) in zip(pred_clses_batch[0].cpu().numpy().astype('int'), pred_bboxes_batch[0].cpu().numpy().astype('int'), pred_quality_scores_batch[0].cpu().numpy()):
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, i2c[cls_id] + ' %.2f' % (score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            for (cls_id, (ymin, xmin, ymax, xmax)) in zip(gt_clses_batch[0].cpu().numpy().astype('int'), gt_bboxes_batch[0].cpu().numpy().astype('int')):
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.putText(image, i2c[cls_id], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            visualize(image, 'valid')













if __name__ == '__main__':
    print('Use GPU: ', use_gpu)
    if use_gpu:
        print('Device name: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    print('Self-training Object Detection\n')
    train()
