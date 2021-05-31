import torch
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss, FeaturesClusteringSeparationLoss, SNNL, \
    DistillationEncoderLoss, DistillationEncoderPrototypesLoss, FeaturesSparsificationLoss, \
    KnowledgeDistillationCELossWithGradientScaling
from utils import get_regularizer
import time
from PIL import Image
from utils.run_utils import *
import numpy as np

class Trainer:
    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None, logdir=None):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.where_to_sim = opts.where_to_sim
        self.step = opts.step
        self.no_mask = opts.no_mask  # if True sequential dataset from https://arxiv.org/abs/1907.13372
        self.overlap = opts.overlap
        self.loss_de_prototypes_sumafter = opts.loss_de_prototypes_sumafter
        self.num_classes = sum(classes) if classes is not None else 0

        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
        else:
            self.old_classes = 0

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # features clustering loss
        self.lfc = opts.loss_fc
        self.lfc_flag = self.lfc > 0.
        # Separation between clustering loss
        self.lfc_sep_clust = opts.lfc_sep_clust
        self.lfc_loss = FeaturesClusteringSeparationLoss(num_classes=sum(classes) if classes is not None else 0,
                                                         logdir=logdir if logdir is not None else '', feat_dim=2048,
                                                         device=self.device, lfc_L2normalized=opts.lfc_L2normalized,
                                                         lfc_nobgr=opts.lfc_nobgr, lfc_sep_clust=self.lfc_sep_clust,
                                                         lfc_sep_clust_ison_proto=opts.lfc_sep_clust_ison_proto,
                                                         orth_sep=opts.lfc_orth_sep, lfc_orth_maxonly=opts.lfc_orth_maxonly)

        # SNNL loss at features space
        self.lSNNL = opts.loss_SNNL
        self.lSNNL_flag = self.lSNNL > 0.
        if classes is not None and logdir is not None:
            self.lSNNL_loss = SNNL(num_classes=sum(classes), logdir=logdir, feat_dim=2048, device=self.device)

        # ILTSS paper loss: http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.pdf
        # https://arxiv.org/abs/1911.03462
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = DistillationEncoderLoss(mask=opts.loss_de_maskedold, loss_de_cosine=opts.loss_de_cosine)

        self.ldeprototype = opts.loss_de_prototypes
        self.ldeprototype_flag = self.ldeprototype > 0.
        self.ldeprototype_loss = DistillationEncoderPrototypesLoss(num_classes=sum(classes) if classes is not None else 0,
                                                                   device=self.device)

        # CIL paper loss: https://arxiv.org/abs/2005.06050
        self.lCIL = opts.loss_CIL
        self.lCIL_flag = self.lCIL > 0. and model_old is not None
        self.lCIL_loss = KnowledgeDistillationCELossWithGradientScaling(temp=1, gs=self.lCIL, device=self.device, norm=False)

        # Features Sparsification Loss
        self.lfs = opts.loss_featspars
        self.lfs_flag = self.lfs > 0.
        self.lfs_loss = FeaturesSparsificationLoss(lfs_normalization=opts.lfs_normalization,
                                                   lfs_shrinkingfn=opts.lfs_shrinkingfn,
                                                   lfs_loss_fn_touse=opts.lfs_loss_fn_touse)

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde or self.lfc or self.lfc_sep_clust or self.lSNNL or self.ldeprototype or \
                                self.lfs or self.lCIL

    def train(self, cur_epoch, optim, train_loader, world_size, scheduler=None, print_int=10, logger=None,
              prototypes=None, count_features=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch + 1, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        lfc = torch.tensor(0.)
        lsep_clusters = torch.tensor(0.)
        lSNNL = torch.tensor(0.)
        ldeprototype = torch.tensor(0.)
        lfs = torch.tensor(0.)
        lCIL = torch.tensor(0.)

        train_loader.sampler.set_epoch(cur_epoch)

        model.train()
        start_time = time.time()
        start_epoch_time = time.time()
        for cur_step, (images, labels) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.lfc_flag or self.lfc_sep_clust
                or self.lSNNL_flag or self.ldeprototype_flag or self.lCIL) \
                    and self.model_old is not None:
                with torch.no_grad():
                    outputs_old, features_old = self.model_old(images, ret_intermediate=self.ret_intermediate)

            optim.zero_grad()
            outputs, features = model(images, ret_intermediate=self.ret_intermediate)

            if self.lfc_flag or self.ldeprototype_flag or self.lfc_sep_clust:
                prototypes, count_features = self._update_running_stats((F.interpolate(
                    input=labels.unsqueeze(dim=1).double(), size=(features['body'].shape[2], features['body'].shape[3]),
                    mode='nearest')).long(), features['body'], self.no_mask, self.overlap, self.step, prototypes,
                                                                        count_features)

            # xxx BCE / Cross Entropy Loss
            if not self.icarl_only_dist:
                loss = criterion(outputs, labels)  # B x H x W
            else:
                loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

            loss = loss.mean()  # scalar

            if self.icarl_combined:
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = outputs_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                              torch.sigmoid(outputs_old))

            # features clustering loss
            if self.lfc_flag or self.lfc_sep_clust:
                lfc, lsep_clusters = self.lfc_loss(labels=labels, outputs=outputs,
                                                   features=features['body'], train_step=cur_step, step=self.step,
                                                   epoch=cur_epoch, incremental_step=self.step, prototypes=prototypes)
            lfc *= self.lfc
            if torch.isnan(lfc):  lfc = torch.tensor(0.)
            lsep_clusters *= self.lfc_sep_clust

            # SNNL loss at features space
            if self.lSNNL_flag:
                lSNNL = self.lSNNL * self.lSNNL_loss(labels=labels, outputs=outputs,
                                                     features=features['body'], train_step=cur_step,
                                                     epoch=cur_epoch)

            # xxx ILTSS (distillation on features or logits)
            if self.lde_flag:
                lde = self.lde * self.lde_loss(features=features['body'], features_old=features_old['body'],
                                               labels=labels, classes_old=self.old_classes)

            if self.lCIL_flag:
                outputs_old_temp = torch.zeros_like(outputs)
                outputs_old_temp[:,:outputs_old.shape[1],:,:] = outputs_old

                lCIL = self.lCIL_loss(outputs=outputs, targets=outputs_old_temp, targets_new=labels)

            if self.ldeprototype_flag:
                ldeprototype = self.ldeprototype * self.ldeprototype_loss(features=features['body'],
                                                                          features_old=features_old[
                                                                              'body'] if self.step != 0 else None,
                                                                          labels=labels,
                                                                          classes_old=self.old_classes,
                                                                          incremental_step=self.step,
                                                                          sequential=self.no_mask,
                                                                          overlapped=self.overlap,
                                                                          outputs_old=outputs_old if self.step != 0 else None,
                                                                          outputs=outputs,
                                                                          loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                                                                          prototypes=prototypes,
                                                                          count_features=count_features)

            # Features Sparsification Loss
            if self.lfs_flag:
                lfs = self.lfs * self.lfs_loss(features=features['body'], labels=labels)

            if self.lkd_flag:
                # resize new output to remove new logits and keep only the old ones
                lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

            # xxx first backprop of previous loss (compute the gradients for regularization methods)
            loss_tot = loss + lkd + lde + l_icarl + lfc + lSNNL + lsep_clusters + ldeprototype + lfs + lCIL

            if self.where_to_sim == 'GPU_server':
                from apex import amp
                with amp.scale_loss(loss_tot, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_tot.backward()

            # xxx Regularizer (EWC, RW, PI)
            if self.regularizer_flag:
                if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows' or distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    if self.where_to_sim == 'GPU_server':
                        with amp.scale_loss(l_reg, optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        l_reg.backward()

            optim.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item() + lfc.item() + lSNNL.item() + lsep_clusters.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item() + lfc.item() + \
                             lSNNL.item() + lsep_clusters.item() + ldeprototype.item() + lfs.item() + lCIL.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch + 1}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Loss={interval_loss}, Time taken={time.time() - start_time}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}, Lfc {lfc}, "
                             f"LSNNL {lSNNL}, Lsepclus {lsep_clusters}, LDEProto {ldeprototype}, Lfeatspars {lfs}, "
                             f"LCIL {lCIL}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Losses/interval_loss', interval_loss, x)
                    if self.lfc_flag:
                        logger.add_scalar('Losses/lfc', lfc.item(), x)
                    if self.lSNNL_flag:
                        logger.add_scalar('Losses/lSNNL', lSNNL.item(), x)
                    if self.lfc_sep_clust:
                        logger.add_scalar('Losses/lsep_clusters', lsep_clusters.item(), x)
                    if self.ldeprototype_flag:
                        logger.add_scalar('Losses/lde_prototypes', ldeprototype.item(), x)
                    if self.lfs_flag:
                        logger.add_scalar('Losses/lfs', lfs.item(), x)
                    if self.lCIL_flag:
                        logger.add_scalar('Losses/lCIL', lCIL.item(), x)


                interval_loss = 0.0
                start_time = time.time()

        logger.info(f"END OF EPOCH {cur_epoch + 1}, TOTAL TIME={time.time() - start_epoch_time}")

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        if not self.where_to_sim == 'GPU_windows':
            torch.distributed.reduce(epoch_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

        if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
            epoch_loss = epoch_loss / world_size / len(train_loader)
            reg_loss = reg_loss / world_size / len(train_loader)
        else:
            if distributed.get_rank() == 0:
                epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch + 1}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss), prototypes, count_features

    def validate(self, loader, metrics, world_size, ret_samples_ids=None, logger=None, vis_dir=None, label2color=None, denorm=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        lfc = torch.tensor(0.)
        lsep_clusters = torch.tensor(0.)
        lSNNL = torch.tensor(0.)
        ldeprototype = torch.tensor(0.)
        lfs = torch.tensor(0.)
        lCIL = torch.tensor(0.)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.ldeprototype_flag or
                    self.lfc_flag or self.lfc_sep_clust or self.lSNNL_flag) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)

                outputs, features = model(images, ret_intermediate=True)

                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                  torch.sigmoid(outputs_old))

                # features clustering loss
                if self.lfc_flag or self.lfc_sep_clust:
                    lfc, lsep_clusters = self.lfc_loss(labels=labels, outputs=outputs,
                                                       features=features['body'], val=True)

                # SNNL loss at features space
                if self.lSNNL_flag:
                    lSNNL = self.lSNNL * self.lSNNL_loss(labels=labels, outputs=outputs,
                                                         features=features['body'], val=True)

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde * self.lde_loss(features=features['body'], features_old=features_old['body'],
                                                   labels=labels, classes_old=self.old_classes)

                # Features Sparsification Loss
                if self.lfs_flag:
                    lfs = self.lfs * self.lfs_loss(features=features['body'], labels=labels, val=True)

                if self.lkd_flag:
                    lkd = self.lkd_loss(outputs, outputs_old)

                if self.lCIL_flag:
                    lCIL = self.lCIL_loss(outputs=outputs, targets=outputs_old, targets_new=labels)

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item() + lfc.item() + lSNNL.item() + \
                            lsep_clusters.item() + ldeprototype.item() + lfs.item() + lCIL.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if vis_dir is not None:
                    image_name = loader.dataset.dataset.dataset.images[i][0].split('/')[1].split('.')[0]
                    image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1,2,0)
                    prediction_tosave = label2color(prediction)[0].astype(np.uint8)
                    label_tosave = label2color(labels)[0].astype(np.uint8)

                    # Image.fromarray(image_tosave).save(f'{vis_dir}/{image_name}_RGB.jpg')
                    Image.fromarray(prediction_tosave).save(f'{vis_dir}/{image_name}_pred.png')
                    # Image.fromarray(label_tosave).save(f'{vis_dir}/{image_name}_label.png')

                    # save also features here
                    if True:
                        try:
                            os.mkdir(f'{vis_dir}/features/')
                        except:
                            pass
                        np.save(f'{vis_dir}/features/{image_name}.npy', features)


                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0],
                                        prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            if not self.where_to_sim == 'GPU_windows':
                torch.distributed.reduce(class_loss, dst=0)
                torch.distributed.reduce(reg_loss, dst=0)

            if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
                class_loss = class_loss / world_size / len(loader)
                reg_loss = reg_loss / world_size / len(loader)
            else:
                if distributed.get_rank() == 0:
                    class_loss = class_loss / distributed.get_world_size() / len(loader)
                    reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])

    def _update_running_stats(self, labels_down, features, sequential, overlapped, incremental_step, prototypes, count_features):
        cl_present = torch.unique(input=labels_down)

        # if overlapped: exclude background as we could not have a reliable statistics
        # if disjoint (not overlapped) and step is > 0: exclude bgr as could contain old classes
        if overlapped or ((not sequential) and incremental_step > 0):
            cl_present = cl_present[1:]

        if cl_present[-1] == 255:
            cl_present = cl_present[:-1]

        features_local_mean = torch.zeros([self.num_classes, 2048], device=self.device)

        for cl in cl_present:
            features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(features.shape[1],-1).detach()
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            # cumulative moving average for each feature vector
            # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                            prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features


