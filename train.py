from __future__ import print_function


import os, json
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix

from datasets import get_dataloader, get_num_classes, get_class_names
from models import get_model

from base_trainer import BaseTrainer
from functools import partial

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.utils import Colorize
from losses import get_criterion, mask_loss_ce

from utils.timer import Timer
from utils.stat_manager import StatManager
from torchvision.utils import save_image as sv

# specific to pytorch-v1 cuda-9.0
# see: https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
# and: https://github.com/pytorch/pytorch/issues/14456
torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
DEBUG = False

def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x

class DecTrainer(BaseTrainer):

    def __init__(self, args, **kwargs):
        super(DecTrainer, self).__init__(args, **kwargs)

        # dataloader
        self.trainloader = get_dataloader(args, cfg, 'train')
        # self.trainloader_val = get_dataloader(args, cfg, 'train_voc')
        self.valloader = get_dataloader(args, cfg, 'val')
        self.denorm = self.trainloader.dataset.denorm
        self.use_triplet = args.use_triplet
        self.loss_3d = args.loss_3d
        self.normalize_feature = args.normalize_feature

        self.nclass = get_num_classes(args)
        self.classNames = get_class_names(args)
        assert self.nclass == len(self.classNames) - 1

        self.classIndex = {}
        for i, cname in enumerate(self.classNames):
            self.classIndex[cname] = i

        # model
        self.enc = get_model(cfg.NET, num_classes=self.nclass)
        self.criterion_cls = get_criterion(cfg.NET.LOSS)

        # optimizer using different LR
        enc_params = self.enc.parameter_groups(cfg.NET.LR, cfg.NET.WEIGHT_DECAY)
        # print((enc_params)[3])
        self.optim_enc = self.get_optim(enc_params, cfg.NET)

        # checkpoint management
        self._define_checkpoint('enc', self.enc, self.optim_enc)
        self._load_checkpoint(args.resume)

        self.fixed_batch = None
        self.fixed_batch_path = args.fixed_batch_path
        if os.path.isfile(self.fixed_batch_path):
            print("Loading fixed batch from {}".format(self.fixed_batch_path))
            self.fixed_batch = torch.load(self.fixed_batch_path)

        # using cuda
        if cfg.NUM_GPUS != 0:
            self.enc = nn.DataParallel(self.enc)
            self.criterion_cls = nn.DataParallel(self.criterion_cls)
            self.enc = self.enc.cuda()
            self.criterion_cls = self.criterion_cls.cuda()

        # CHANGE: visual
        self.visual_times = 0
        self.dataset = args.dataset.lower()

    def step(self, epoch, image, gt_labels, train=False, visualise=False, save_image=False, info=None):

        PRETRAIN = epoch < (11 if DEBUG else cfg.TRAIN.PRETRAIN)

        if self.dataset == "wikiscenes_corr":
            corr = image["corr"]
            image = image["image"]
        
        # denorm image
        image_raw = self.denorm(image.clone())

        # classification
        cls_out, cls_fg, masks, mask_logits, pseudo_gt, loss_mask = self.enc(image, image_raw, gt_labels)
        
        # classification loss
        loss_cls = self.criterion_cls(cls_out, gt_labels).mean()
        
        # keep track of all losses for logging
        losses = {"loss_cls": loss_cls.item()}
        losses["loss_fg"] = cls_fg.mean().item()

        loss_3d = 0
        mean_3d_loss = 0
        kp_number = 0
        norm = 0
        if self.dataset == "wikiscenes_corr" and train:
            # compute 3D consistency loss
            feature = masks["feature"]
            # here we normalize the out_feature to make sure it doesn't optimize by scaling the feature vector
            if self.normalize_feature:
                feature = feature / feature.norm(dim=1, keepdim=True)
            b, c, h, w = feature.shape
            feature = feature.reshape(2, b // 2, c, h, w)
            assert h == w, "not square"
            for i in range(b // 2):
                if not corr[i]:
                    continue
                # k * 4
                coord = torch.tensor(corr[i])
                if cfg.NUM_GPUS != 0:
                    coord = coord.cuda()
                # reshape, permute to allow grid_sample, [N, Hout, Wout, 2]
                coord = coord.reshape(1, -1, 2, 2).permute(2, 0, 1, 3).contiguous()
                if self.use_triplet:
                    # add a fake kp
                    coord = torch.cat([coord, coord[1:2]])
                    # add a random shift in [0.25, 0.75], fmod it
                    # so the selected fake keypoint is far enough from the correct corresponding one
                    coord[2,0,:,:] += (torch.randn(coord.shape[2], 2) / 2 - 0.25).cuda()
                    coord = coord.fmod(1.0)
                    # modify feature so that it has the same batch size as coord
                    modified_feature = torch.cat([feature, feature[1:]])
                else:
                    modified_feature = feature
                # change range to [-1, 1]
                coord = coord * 2 - 1

                keypoints = F.grid_sample(modified_feature[:,i,:,:], coord)
                loss_func = nn.MSELoss()
                with torch.no_grad():
                    mean_3d_loss += loss_func(feature[:,0,:,:], feature[:,1,:,:])
                    norm += torch.norm(feature).item()
                if self.use_triplet:
                    distance_p = (keypoints[0] - keypoints[1]).norm(dim=0)
                    distance_n = (keypoints[0] - keypoints[2]).norm(dim=0)
                    loss_3d += nn.ReLU()(args.triplet_margin + distance_p - distance_n).mean()
                else:
                    loss_3d += loss_func(keypoints[0], keypoints[1])
                kp_number += coord.shape[2]

            losses["loss_3d"] = loss_3d
            losses["mean_loss_3d"] = mean_3d_loss
            losses["feature_norm"] = norm
            losses["kp number"] = kp_number
        
        loss = loss_cls.clone()
        if "dec" in masks:
            loss_mask = loss_mask.mean()

            # (CHANGE: disable mask loss, only cls loss)
            if not PRETRAIN:
                loss += cfg.NET.MASK_LOSS_BCE * loss_mask

            assert not "pseudo" in masks
            masks["pseudo"] = pseudo_gt
            losses["loss_mask"] = loss_mask.item()

        # add 3d consistency loss
        if self.dataset == "wikiscenes_corr" and train:
            loss += losses["loss_3d"] * self.loss_3d

        losses["loss"] = loss.item()

        if train:
            self.optim_enc.zero_grad()
            loss.backward()
            self.optim_enc.step()

        for mask_key, mask_val in masks.items():
            masks[mask_key] = masks[mask_key].detach()

        mask_logits = mask_logits.detach()

        if visualise:
            self._visualise(epoch, image, masks, mask_logits, cls_out, gt_labels, save_image, info)

        # make sure to cut the return values from graph
        return losses, cls_out.detach(), masks, mask_logits

    def train_epoch(self, epoch):
        self.enc.train()

        stat = StatManager()
        stat.add_val("loss")
        stat.add_val("loss_cls")
        stat.add_val("loss_fg")
        stat.add_val("loss_bce")

        # adding stats for classes
        timer = Timer("New Epoch: ")
        # train_step = partial(self.step, train=True, visualise=False)
        # CHANGE: get result, not train
        train_step = partial(self.step, train=True, visualise=False)

        preds_all = list()
        targets_all = list()

        for i, (image, gt_labels, _) in enumerate(self.trainloader):

            if self.dataset == "wikiscenes_corr":
                corr = image['corr']
                for i in range(len(corr)):
                    corr[i] = json.loads(corr[i])
                image = torch.cat([image['1'], image['2']], 0)
                image_corr = {"image": image, "corr": corr}
                gt_labels = torch.cat(gt_labels, 0)

                losses, cls_out, _, _ = train_step(epoch, image_corr, gt_labels)
            else:
                losses, cls_out, _, _ = train_step(epoch, image, gt_labels)

            cls_sigmoid = torch.sigmoid(cls_out.cpu()).numpy()
            preds_all.append(cls_sigmoid)
            targets_all.append(gt_labels.cpu().numpy())


            if self.fixed_batch is None:
                self.fixed_batch = {}
                self.fixed_batch["image"]   = image.clone()
                self.fixed_batch["labels"]  = gt_labels.clone()
                torch.save(self.fixed_batch, self.fixed_batch_path)

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            # intermediate logging
            if i % 10 == 0:
                msg =  "Loss [{:04d}]: ".format(i)
                for loss_key, loss_val in losses.items():
                    msg += "{}: {:.4f} | ".format(loss_key, loss_val)
                
                msg += " | Im/Sec: {:.1f}".format(i * cfg.TRAIN.BATCH_SIZE / timer.get_stage_elapsed())
                print(msg)
                sys.stdout.flush()

            del image, gt_labels

            if DEBUG and i > 0:
                break

        def publish_loss(stats, name, t, prefix='data/'):
            print("{}: {:4.3f}".format(name, stats.summarize_key(name)))
            self.writer.add_scalar(prefix + name, stats.summarize_key(name), t)

        for stat_key in stat.vals.keys():
            publish_loss(stat, stat_key, epoch)

        # plotting learning rate
        for ii, l in enumerate(self.optim_enc.param_groups):
            print("Learning rate [{}]: {:4.3e}".format(ii, l['lr']))
            self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

        # self.writer.add_scalar('lr/bg_baseline', self.enc.module.mean.item(), epoch)

        self.count_acc(targets_all, preds_all, self.writer, epoch)

        # visualising
        # self.enc.eval()
        # with torch.no_grad():
        #     self.step(epoch, self.fixed_batch["image"], \
        #                      self.fixed_batch["labels"], \
        #                      train=False, visualise=True)

        if self.dataset == 'wikiscenes_corr':
            

    def _mask_rgb(self, masks, image_norm):
        # visualising masks
        masks_conf, masks_idx = torch.max(masks, 1)
        masks_conf = masks_conf - F.relu(masks_conf - 1, 0)

        masks_idx_rgb = self._apply_cmap(masks_idx.cpu(), masks_conf.cpu())
        return 0.3 * image_norm + 0.7 * masks_idx_rgb

    def _init_norm(self):
        self.trainloader.dataset.set_norm(self.enc.normalize)
        self.valloader.dataset.set_norm(self.enc.normalize)
        # self.trainloader_val.dataset.set_norm(self.enc.normalize)

    def _apply_cmap(self, mask_idx, mask_conf):
        palette = self.trainloader.dataset.get_palette()

        masks = []
        col = Colorize()
        mask_conf = mask_conf.float() / 255.0
        for mask, conf in zip(mask_idx.split(1), mask_conf.split(1)):
            m = col(mask).float()
            m = m * conf
            masks.append(m[None, ...])

        return torch.cat(masks, 0)

    def validation(self, epoch, writer, loader, checkpoint=False):

        stat = StatManager()

        # Fast test during the training
        def eval_batch(image, gt_labels, info):

            losses, cls, masks, mask_logits = \
                    self.step(epoch, image, gt_labels, train=False, visualise=True, save_image=True, info=info)

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            return cls.cpu(), masks, mask_logits.cpu()

        self.enc.eval()

        # class ground truth
        targets_all = []

        # class predictions
        preds_all = []

        def add_stats(means, stds, x):
            means.append(x.mean())
            stds.append(x.std())

        for n, (image, gt_labels, info) in enumerate(loader):

            if self.dataset == "wikiscenes_corr":
                # info = info[0] + info[1]
                info = info[0]
                corr = image['corr']
                for i in range(len(corr)):
                    corr[i] = json.loads(corr[i])
                # image = torch.cat([image['1'], image['2']], 0)
                # not validate the random selected ones
                image = image['1']
                image_corr = {"image": image, "corr": corr}
                # gt_labels = torch.cat(gt_labels, 0)
                gt_labels = gt_labels[0]

            with torch.no_grad():
                cls_raw, masks_all, mask_logits = eval_batch(
                    image_corr if self.dataset == "wikiscenes_corr" else image, 
                    gt_labels, info)

            cls_sigmoid = torch.sigmoid(cls_raw).numpy()

            preds_all.append(cls_sigmoid)
            targets_all.append(gt_labels.cpu().numpy())

        self.count_acc(targets_all, preds_all, writer, epoch)

        # total classification loss
        for stat_key in stat.vals.keys():
            writer.add_scalar('all/{}'.format(stat_key), stat.summarize_key(stat_key), epoch)

        if checkpoint and epoch >= cfg.TRAIN.PRETRAIN: 
            # we will use mAP - mask_loss as our proxy score
            # to save the best checkpoint so far
            proxy_score = 1 - stat.summarize_key("loss")
            writer.add_scalar('all/checkpoint_score', proxy_score, epoch)
            self.checkpoint_best(proxy_score, epoch)

    def count_acc(self, targets_all, preds_all, writer, epoch):

        #
        # classification
        #
        targets_stacked = np.vstack(targets_all)
        preds_stacked = np.vstack(preds_all)
        aps = average_precision_score(targets_stacked, preds_stacked, average=None)

        # y_true = targets_stacked.argmax(1)
        # y_pred = preds_stacked.argmax(1)
        # acc = accuracy_score(y_true, y_pred)

        # per class accuracy
        # give up accuracy since computationally intensive
        # cm = confusion_matrix(y_true, y_pred)
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm = cm.diagonal()

        # skip BG AP
        offset = self.nclass - aps.size
        assert offset == 1, 'Class number mismatch'

        classNames = self.classNames[offset:-1]
        for ni, className in enumerate(classNames):
            writer.add_scalar('%02d_%s/AP' % (ni + offset, className), aps[ni], epoch)
            # writer.add_scalar('%02d_%s/acc' % (ni + offset, className), cm[ni], epoch)
            print("AP_{}: {:4.3f}".format(className, aps[ni]))

        meanAP = np.mean(aps)
        writer.add_scalar('all_wo_BG/mAP', meanAP, epoch)
        # writer.add_scalar('all_wo_BG/acc', acc, epoch)
        print('mAP: {:4.3f}'.format(meanAP))

    def _visualise(self, epoch, image, masks, mask_logits, cls_out, gt_labels, for_save=False, info=None):
        image_norm = self.denorm(image.clone()).cpu()
        visual = [image_norm]

        if "cam" in masks:
            visual.append(self._mask_rgb(masks["cam"], image_norm))


        if "dec" in masks:
            visual.append(self._mask_rgb(masks["dec"], image_norm))

        if "pseudo" in masks:
            pseudo_gt_rgb = self._mask_rgb(masks["pseudo"], image_norm)

            # cancel ambiguous
            ambiguous = 1 - masks["pseudo"].sum(1, keepdim=True).cpu()
            pseudo_gt_rgb = ambiguous * image_norm + (1 - ambiguous) * pseudo_gt_rgb
            visual.append(pseudo_gt_rgb)

        if "full" in masks:
            bg = masks["full"][:,0,:,:]
            for i in range(1, self.nclass):
                temp = torch.zeros_like(masks["full"])
                temp[:,0,:,:] = bg
                temp[:,i,:,:] = masks["full"][:,i,:,:]
                visual.append(self._mask_rgb(temp, image_norm))

        # ready to assemble
        visual_logits = torch.cat(visual, -1)
        # self._visualise_grid(visual_logits, gt_labels, epoch, scores=cls_out)
        # CHANGE: visualize
        self._visualise_grid(visual_logits, gt_labels, epoch, scores=cls_out, save_image=for_save, epoch=epoch, index=self.visual_times, info=info)
        if for_save:
            self.visual_times += 1

if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Config: \n", cfg)

    trainer = DecTrainer(args)
    torch.manual_seed(0)

    timer = Timer()
    def time_call(func, msg, *args, **kwargs):
        timer.reset_stage()
        func(*args, **kwargs)
        print(msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))

    for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        print("Epoch >>> ", epoch)

        log_int = 5 if DEBUG else 2
        if epoch % log_int == 0:
            with torch.no_grad():
                # if not DEBUG:
                #     time_call(trainer.validation, "Validation / Train: ", epoch, trainer.writer, trainer.trainloader_val)
                time_call(trainer.validation, "Validation / Val: ", epoch, trainer.writer_val, trainer.valloader, checkpoint=True)

        time_call(trainer.train_epoch, "Train epoch: ", epoch)
