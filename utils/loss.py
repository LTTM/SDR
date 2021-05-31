import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis, NMF, TruncatedSVD, LatentDirichletAllocation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
from MulticoreTSNE import MulticoreTSNE
import os

def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


# CIL loss defined in paper: https://arxiv.org/abs/2005.06050
# This loss was provided by the authors of the paper.
class KnowledgeDistillationCELossWithGradientScaling(nn.Module):
    def __init__(self, temp=1, device=None, gs=1, norm=False):
        """Initialises the loss

                :param temp: temperature of the knowledge distillation loss, reduces to CE-loss for t = 1
                :param device: torch device used during training
                :param gs: defines the strength of the scaling
                :param norm: defines how the loss is normalized

        """

        super().__init__()
        assert isinstance(temp, int), "temp has to be of type int, default is 1"
        assert isinstance(device, torch.device), "device has to be of type torch.device"
        # assert gs > 0, "gs has to be > 0"
        assert isinstance(norm, bool), "norm has to be of type bool"

        self.temp = temp
        self.device = device
        self.gs = gs
        self.norm = norm

    def forward(self, outputs, targets, targets_new=None):
        assert torch.is_tensor(outputs), "outputs has to be of type torch.tensor"
        assert torch.is_tensor(targets), "targets has to be of type torch.tensor"
        assert outputs.shape == targets.shape, "shapes of outputs and targets have to agree"
        assert torch.is_tensor(targets_new) or targets_new is None, \
            "targets_new may only be of type torch.tensor or 'None'"

        """forward function

                        output: output of the network
                        targets: soft targets from the teacher
                        targets_new: hard targets for the new classes

                """
        # Set probabilities to 0 for pixels that belong to new classes, i. e. no knowledge is distilled for pixels
        # having hard labels
        # outputs       = B x C x d_1 x d_2 x ...
        # targets       = B x C x d_1 x d_2 x ...
        # targets_new   = B x d_1 x d_2 x ...
        # mask          = B x d_1 x d_2 x ...

        # here the weights are calculated as described in the paper, just remove the weights from the calculation as
        # in KnowledgeDistillationCELoss
        targets = torch.softmax(targets, dim=1)
        outputs = torch.softmax(outputs, dim=1)
        denom_corr = 0
        ln2 = torch.log(torch.tensor([2.0]).to(self.device))  # basis change
        entropy = -torch.sum(targets * torch.log(targets+1e-8), dim=1, keepdim=True) / ln2
        weights = entropy * self.gs + 1

        # calculate the mask from the new targets, so that only the regions without labels are considered
        if targets_new is not None:
            mask = torch.zeros(targets_new.shape).to(self.device)
            mask[targets_new == 255] = 1
            mask[targets_new == 0] = 1
            denom_corr = torch.numel(mask) - int(torch.sum(mask))
            mask = mask.reshape(shape=(mask.shape[0], 1, *mask.shape[1:]))
            weights = mask * weights
            mask = mask.expand_as(targets)
            targets = mask * targets

        # Calculate unreduced loss
        loss = weights * torch.sum(targets * outputs, dim=1, keepdim=True)

        # Apply mean reduction
        if self.norm:
            denom = torch.sum(weights)
        else:
            denom = torch.numel(loss[:, 0, ...]) - denom_corr
        loss = torch.sum(loss) / (denom+1e-8)

        return self.temp**2 * loss  # Gradients are scaled down by 1 / T if not corrected


class SNNL(nn.Module):
    def __init__(self, device, reduction='mean', alpha=1., num_classes=0, logdir=None, feat_dim=2048):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.num_classes = num_classes
        self.device = device
        self.logdir = logdir

    def forward(self, labels, outputs, features, train_step, epoch, val=False, mask=None):
        loss = torch.tensor(0., device=self.device)
        temperature = 1

        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()

        x = features.view(-1, features.shape[1])  # bF*2048
        y = labels_down.view(-1)  # bF

        cl_present = torch.unique(input=labels_down)

        r = 0
        for i in range(x.shape[0]):
            numerator = denominator = 0
            xi = x[i,:]
            for j in range(x.shape[0]):
                xj = x[j,:]
                if j != i:
                    if y[i] == y[j]:
                        numerator += torch.exp(-torch.norm(xi-xj)**2 / temperature)

                    denominator += torch.exp(-torch.norm(xi-xj)**2 / temperature)

            r += (torch.log(numerator/denominator))

        loss = - r/x.shape[0]

        return loss


# Contrastive Learning loss defined in SDR: https://arxiv.org/abs/2103.06342
class FeaturesClusteringSeparationLoss(nn.Module):
    def __init__(self, device, reduction='mean', alpha=1., num_classes=0, logdir=None, feat_dim=2048,
                 lfc_L2normalized=False, lfc_nobgr=False, lfc_sep_clust=0., lfc_sep_clust_ison_proto=False,
                 orth_sep=False, lfc_orth_maxonly=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.num_classes = num_classes
        self.device = device
        self.logdir = logdir
        self.lfc_L2normalized = lfc_L2normalized
        self.lfc_nobgr = lfc_nobgr
        self.lfc_sep_clust = lfc_sep_clust
        self.lfc_sep_clust_ison_proto = lfc_sep_clust_ison_proto
        self.orth_sep = orth_sep
        self.lfc_orth_maxonly = lfc_orth_maxonly
        self.feat_dim = feat_dim

    def _visualize_with_tSNE(self, tSNE_path_to_save, features, labels_down, epoch=0, train_step=0, step=0):
        # visualization with t-SNE from sklearn (very slow, can improve 1000x with t-SNE for CUDA
        # or x100 with t-SNE w multiprocessing
        # X_embedded = TSNE(n_components=2).fit_transform(features.detach().view(-1, features.shape[1]))
        classes = {
            0: 'background',
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
            10: 'cow',
            11: 'diningtable',
            12: 'dog',
            13: 'horse',
            14: 'motorbike',
            15: 'person',
            16: 'pottedplant',
            17: 'sheep',
            18: 'sofa',
            19: 'train',
            20: 'tvmonitor'
        }
        tsne = MulticoreTSNE(n_jobs=4)

        features = features.detach().view(-1, features.shape[1]).cpu()
        labels_down = labels_down.view(-1)
        features_nobgr = features[labels_down != 0, :]
        labels_down = labels_down[labels_down != 0]

        features_nobgr = F.normalize(features_nobgr, p=2, dim=0)

        X_embedded_multicore = tsne.fit_transform(features_nobgr)

        plt.figure()
        sns.scatterplot(X_embedded_multicore[:, 0], X_embedded_multicore[:, 1], hue=[classes[x] for x in labels_down.view(-1).cpu().numpy()],
                        legend='full', palette=sns.color_palette("bright", torch.unique(labels_down).size(0)))
        plt.savefig(f"{tSNE_path_to_save}/step_{step}_epoch_{epoch}_{train_step}_tSNE.png")
        plt.close()

    def forward(self, labels, outputs, features, train_step=0, step=0, epoch=0, val=False, mask=None, prototypes=None,
                incremental_step=None):
        loss_features_clustering = torch.tensor(0., device=self.device)
        loss_separationclustering = torch.tensor(0., device=self.device)

        if (not val) and (incremental_step != 0):
            labels = labels.unsqueeze(dim=1)
            labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()
            cl_present = torch.unique(input=labels_down)

            if self.lfc_nobgr:
                cl_present = cl_present[1:]

            if cl_present[-1] == 255:
                cl_present = cl_present[:-1]

            features_local_mean = torch.zeros([self.num_classes, self.feat_dim], device=self.device)

            for cl in cl_present:
                features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(features.shape[1], -1)

                # L2 normalization of the features
                if self.lfc_L2normalized:
                    features_cl = F.normalize(features_cl, p=2, dim=0)
                    prototypes = F.normalize(prototypes, p=2, dim=0)

                features_local_mean[cl] = torch.mean(features_cl, dim=-1)

                loss_to_use = nn.MSELoss()
                loss_features_clustering += loss_to_use(features_cl, prototypes[cl].unsqueeze(1).expand(-1, features_cl.shape[1]))

                loss_features_clustering /= (cl_present.shape[0])

            if self.lfc_sep_clust > 0.:
                features_local_mean_reduced = features_local_mean[cl_present,:]  # remove zero rows
                if not self.orth_sep:
                    if not self.lfc_sep_clust_ison_proto:
                        inv_pairwise_D = 1 / torch.cdist(features_local_mean_reduced.unsqueeze(dim=0),
                                                         features_local_mean_reduced.unsqueeze(dim=0)).squeeze()
                    else:
                        inv_pairwise_D = 1 / torch.cdist(features_local_mean_reduced.unsqueeze(dim=0),
                                                         prototypes.detach()[features_local_mean.abs().sum(dim=-1) != 0].unsqueeze(dim=0)).squeeze()

                    loss_separationclustering_temp = inv_pairwise_D[~torch.isinf(inv_pairwise_D)].mean()
                    if ~torch.isnan(loss_separationclustering_temp): loss_separationclustering = loss_separationclustering_temp
                else:
                    # features in features_local_mean_reduced to be orthogonal to those in self if not belongin to the same class

                    vectorial_products = (torch.mm(prototypes, features_local_mean_reduced.T)).squeeze()
                    vectorial_products[cl_present, range(0, cl_present.shape[0])] = 0

                    if self.lfc_orth_maxonly:
                        loss_separationclustering = torch.max(vectorial_products)
                    else:
                        loss_separationclustering = torch.sum(vectorial_products) / \
                                                    (vectorial_products.shape[0]*vectorial_products.shape[1] -
                                                     cl_present.shape[0])

        # visualize every epoch
        img_path_to_save = self.logdir
        if not val and not(os.path.exists(f"{img_path_to_save}/step_{step}_epoch_{epoch}_{train_step}_tSNE.png"))and train_step % 250 == 0 and (incremental_step != 0):
            os.makedirs(img_path_to_save, exist_ok=True)
            self._visualize_with_tSNE(img_path_to_save, features, labels_down, epoch, train_step, step)
        return loss_features_clustering, loss_separationclustering


class DistillationEncoderLoss(nn.Module):
    def __init__(self, mask=False, loss_de_cosine=False):
        super().__init__()
        self.mask = mask
        self.loss_de_cosine = loss_de_cosine

    def _compute_mask_old_classes(self, features, labels, classes_old):
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest'))
        mask = labels_down < classes_old
        return mask


    def forward(self, features, features_old, labels, classes_old):
        if not self.loss_de_cosine:
            loss_to_use = nn.MSELoss(reduction='none')
            loss = loss_to_use(features, features_old)
            if self.mask:
                masked_features = self._compute_mask_old_classes(features, labels, classes_old)
                loss = loss[masked_features.expand_as(loss)]
        else:
            loss_to_use = nn.CosineSimilarity()
            loss = 1.0 - loss_to_use(features, features_old)
            if self.mask:
                masked_features = self._compute_mask_old_classes(features, labels, classes_old)
                loss = loss[masked_features.squeeze()]

        outputs = torch.mean(loss)

        return outputs


# Prototypes Matching loss defined in SDR: https://arxiv.org/abs/2103.06342
class DistillationEncoderPrototypesLoss(nn.Module):
    def __init__(self, device, num_classes, mask=False):
        super().__init__()
        self.mask = mask
        self.num_classes = num_classes
        self.device = device

    def _compute_mask_old_classes(self, features, labels, classes_old):
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest'))
        mask = labels_down < classes_old
        return mask

    def forward(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None):
        outputs = torch.tensor(0., device=self.device)
        MSEloss_to_use = nn.MSELoss()
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()
        labels_down_bgr_mask = (labels_down == 0).long()

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)    # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()

                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    cl_present = cl_present[1:]

                for cl in cl_present:
                    prototype = prototypes.detach()[cl]
                    current_features = features[(pseudolabel_old_down == cl).expand_as(features)].view(-1, features.shape[1])

                    if loss_de_prototypes_sumafter:
                        current_proto = torch.mean(current_features, dim=0)
                        outputs += MSEloss_to_use(current_proto, prototype) / cl_present.shape[0]
                    else:
                        for f in range(current_features.size(0)):
                            outputs += MSEloss_to_use(current_features[f, :], prototype) / (current_features.shape[0])

        return outputs


# Features Sparsification loss defined in SDR: https://arxiv.org/abs/2103.06342
class FeaturesSparsificationLoss(nn.Module):
    def __init__(self, lfs_normalization, lfs_shrinkingfn, lfs_loss_fn_touse, mask=False, reduction='mean'):
        super().__init__()
        self.mask = mask
        self.lfs_normalization = lfs_normalization
        self.lfs_shrinkingfn = lfs_shrinkingfn
        self.lfs_loss_fn_touse = lfs_loss_fn_touse
        self.eps = 1e-15
        self.reduction = reduction

    def forward(self, features, labels, val=False):
        outputs = torch.tensor(0.)

        if not val:
            labels = labels.unsqueeze(dim=1)
            labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()

            if self.lfs_normalization == 'L1':
                features_norm = F.normalize(features, p=1, dim=1)
            elif self.lfs_normalization == 'L2':
                features_norm = F.normalize(features, p=2, dim=1)
            elif self.lfs_normalization == 'max_foreachfeature':
                features_norm = features / (torch.max(features, dim=1, keepdim=True).values + self.eps)
            elif self.lfs_normalization == 'max_maskedforclass':
                features_norm = torch.zeros_like(features)
                classes = torch.unique(labels_down)
                if classes[-1] == 0:
                    classes = classes[:-1]
                for cl in classes:
                    cl_mask = labels_down == cl
                    features_norm += (features / (torch.max(features[cl_mask.expand(-1, features.shape[1], -1, -1)]) + self.eps)) * cl_mask.float()
            elif self.lfs_normalization == 'max_overall':
                features_norm = features / (torch.max(features) + self.eps)
            elif self.lfs_normalization == 'softmax':
                features_norm = torch.softmax(features, dim=1)

            if features_norm.sum() > 0:
                if self.lfs_shrinkingfn == 'squared':
                    shrinked_value = torch.sum(features_norm**2, dim=1, keepdim=True)
                if self.lfs_shrinkingfn == 'power3':
                    shrinked_value = torch.sum(features_norm ** 3, dim=1, keepdim=True)
                elif self.lfs_shrinkingfn == 'exponential':
                    shrinked_value = torch.sum(torch.exp(features_norm), dim=1, keepdim=True)

                summed_value = torch.sum(features_norm, dim=1, keepdim=True)

                if self.lfs_loss_fn_touse == 'ratio':
                    outputs = shrinked_value / (summed_value + self.eps)
                elif self.lfs_loss_fn_touse == 'lasso':  # NB: works at features space directly
                    outputs = torch.norm(features, 1) / 2  # simple L1 (Lasso) regularization
                elif self.lfs_loss_fn_touse == 'max_minus_ratio':
                    # TODO: other loss functions to be considered
                    # outputs = summed_value - shrinked_value / summed_value
                    pass
                elif self.lfs_loss_fn_touse == 'entropy':  # NB: works only with probabilities (i.e. with L1 or softmax as normalization)
                    outputs = torch.sum(- features_norm * torch.log(features_norm + 1e-10), dim=1)

        if self.reduction == 'mean':
            return outputs.mean()
        elif self.reduction == 'sum':
            return outputs.sum()


