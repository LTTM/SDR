import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

from functools import partial, reduce

import models
from modules import DeeplabV3
import net.resnet_atrous as atrousnet
import net.xception as xception

def make_model(opts, classes=None):
    import inplace_abn
    from inplace_abn import InPlaceABNSync, InPlaceABN, ABN
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')

        # remove 'module' prefix from pre_dict
        pre_dict = _remove_module_prefix(pre_dict)

        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    head_channels = 256

    head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                     out_stride=opts.output_stride, pooling_size=opts.pooling)

    if classes is not None:
        model = IncrementalSegmentationModule(body, head, head_channels, classes=classes, fusion_mode=opts.fusion_mode, net_pytorch=opts.net_pytorch)
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)

    return model


def make_model_v2(opts, classes=None):
    '''
    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)'''
    net_name = opts.backbone.replace('net','') + '_atrous'
    body = _build_backbone(backbone_name=net_name, pretrained=not opts.no_pretrained)

    #del pre_dict['fc.weight']  # pre_dict['state_dict']['classifier.fc.weight']
    #del pre_dict['fc.bias']  # pre_dict['state_dict']['classifier.fc.bias']

    head_channels = 256

    head = DeeplabV3(body.inplanes, head_channels, 256, out_stride=opts.output_stride, pooling_size=opts.pooling)

    if classes is not None:
        model = IncrementalSegmentationModule(body, head, head_channels, classes=classes, fusion_mode=opts.fusion_mode, net_pytorch=opts.net_pytorch)
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)

    return model





def _build_backbone(backbone_name='res101_atrous', pretrained=True, os=16):
    if backbone_name == 'res50_atrous':
        net = atrousnet.resnet50_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'res101_atrous':
        net = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'res152_atrous':
        net = atrousnet.resnet152_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'xception' or backbone_name == 'Xception':
        net = xception.xception(pretrained=pretrained, os=os)
        return net
    else:
        raise ValueError('backbone.py: The backbone named %s is not supported yet.' % backbone_name)


def _remove_module_prefix(pre_dict):
    pre_dict['state_dict'] = {key.replace('module.', ''): value for key, value in pre_dict['state_dict'].items()}
    return pre_dict


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes, net_pytorch=True, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = nn.ModuleList(
            [nn.Conv2d(head_channels, c, 1) for c in classes]
        )
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None
        self.net_pytorch = net_pytorch

    def _network(self, x, ret_intermediate=False):

        x_b = self.body(x)
        x_pl = self.head(x_b)
        out = []
        for mod in self.cls:
            out.append(mod(x_pl))
        x_o = torch.cat(out, dim=1)

        if ret_intermediate:
            return x_o, x_b, x_pl
        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate)

        sem_logits = out[0] if ret_intermediate else out

        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        if ret_intermediate:
            return sem_logits, {"body": out[1], "pre_logits": out[2]}

        return sem_logits, {}

    def fix_bn(self):
        if self.net_pytorch:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        else:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

