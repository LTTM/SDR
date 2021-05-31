import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP

class BSANet(nn.Module):
	def __init__(self, cfg):
		super(BSANet, self).__init__()
		self.backbone = None
		self.backbone_layers = None
		input_channel = 2048
		self.aspp = ASPP(dim_in=input_channel,
				dim_out=cfg.MODEL_ASPP_OUTDIM,
				rate=16//cfg.MODEL_OUTPUT_STRIDE,
				bn_mom = cfg.TRAIN_BN_MOM)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)
		self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)


		# set the dim of ASPP output, default set as 256 for memory limitation
		indim = 256
		asppdim=256


		self.edge_conv = nn.Sequential(
				nn.Conv2d(indim, indim, 1, 1,
						  padding=0, bias=True),
				SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),

				nn.Conv2d(indim, indim // 2, 3, 1,
						  padding=1, bias=True),
				SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),

				nn.Conv2d(indim // 2, indim // 2, 1, 1,
						  padding=0, bias=True),
				SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
		)

		self.edge_outconv=nn.Conv2d(indim //2,2, kernel_size=1, padding=0, dilation=1, bias=True)

		self.edge_up=nn.UpsamplingBilinear2d(scale_factor=4)

		self.sigmoid = nn.Sigmoid()

		##mid level encoder
		self.midedge_conv = nn.Sequential(
			nn.Conv2d(indim*2, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim // 2, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim // 2, indim // 2, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
		)

		self.shortcut_conv_mid = nn.Sequential(
			nn.Conv2d(indim*2, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim // 2, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim // 2, indim // 2, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim // 2, momentum=cfg.TRAIN_BN_MOM),
		)

		self.shortcut_conv_high = nn.Sequential(
			nn.Conv2d(indim*4, indim, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL // 2,
					  bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)

		self.highedge_conv = nn.Sequential(
			nn.Conv2d(indim*4, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
		)

		self.highedge_outconv = nn.Conv2d(indim, 2, kernel_size=1, padding=0, dilation=1, bias=True)


		self.midedge_outconv = nn.Conv2d(indim // 2, 2, kernel_size=1, padding=0, dilation=1, bias=True)

		self.midedge_up = nn.UpsamplingBilinear2d(scale_factor=8)

		## low-level feature transformation 
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim  , indim //2, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
				SynchronizedBatchNorm2d(indim //2, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
		)


		self.query = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv2d(indim, indim//2, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
			SynchronizedBatchNorm2d(indim//2, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)


		#
		self.cat_conv1 = nn.Sequential(
			nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + indim // 2, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0, bias=True),
			SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)
		self.relu=nn.ReLU(inplace=True)


		## semantic encoder with 21 classes
		self.semantic_encoding = nn.Sequential(
			nn.Conv2d(asppdim, indim, 1, 1,
					  padding=0, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),

			nn.Conv2d(indim, indim, 3, 1,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
		)

		self.avg_pool=nn.AdaptiveAvgPool2d(1)
		self.semantic_fc = nn.Sequential(
			nn.Conv2d(indim, asppdim, 3, 2,
					  padding=1, bias=True),
			SynchronizedBatchNorm2d(asppdim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)


		self.fc = nn.Sequential(
			nn.Linear(asppdim, asppdim // 4,bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(asppdim //4, asppdim, bias=False),
			nn.Sigmoid(),
		)


		self.semantic_output = nn.Conv2d(indim, 21, 1, 1, padding=0)

		self.upsample_conv = nn.Sequential(
			nn.Conv2d(indim, indim, 1, 1, padding=1 // 2,
					  bias=True),
			SynchronizedBatchNorm2d(indim, momentum=cfg.TRAIN_BN_MOM),
			nn.ReLU(inplace=True),
		)
		##cfg.MODEL_SHORTCUT_DIM
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+indim, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)

		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x,mode):

		# mode: option "train" "test"
		# This code is simplified to get readability from the original implementation

		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)

		feature_0 = self.shortcut_conv(layers[1])
		feature_1 = self.shortcut_conv_mid(layers[2])
		feature_2 = self.shortcut_conv_high(layers[3])

		edge2 = self.highedge_conv(layers[3])

		edge2_r = self.highedge_outconv(edge2)

		feature_2 = torch.mul(feature_2, edge2)

		feature_2 = self.query(feature_2)

		feature_cat = torch.cat([feature_aspp,feature_2],1)

		feature_cat = self.cat_conv1(feature_cat)

		feature_edge = self.edge_conv(layers[1])

		edge = self.edge_outconv(feature_edge)
		edge_r = self.edge_up(edge)

		attention_edg= self.sigmoid(feature_edge)

		feature_low=torch.mul(feature_0, feature_edge)

		##mid
		feature_edge1 = self.midedge_conv(layers[2])
		edge1 = self.midedge_outconv(feature_edge1)

		edge1_r = self.midedge_up(edge1)

		b, c, h, w = edge1.size()

		attention_edg1 = self.sigmoid(feature_edge1)
		feature_mid = torch.mul(feature_1, attention_edg1)

		feature_mid = self.upsample2(feature_mid)

		feature_cat = self.upsample_sub(feature_cat)
		feature_cat = torch.cat([feature_cat,feature_mid,feature_low],1)


		feature_cat = self.cat_conv(feature_cat)

		b, c, _, _ = feature_cat.size()

		#### semantic encoding


		feature_semantic = self.semantic_encoding(feature_cat)



		fc_att = self.semantic_fc(feature_semantic)
		fc_att = self.avg_pool(fc_att).view(b,c)
		fc_att = self.fc(fc_att)
		# use dense attention to get a little higher performance boost +0.30% miou

		fc_att= fc_att.view(b, c, 1, 1)
		feature_final = F.relu_(feature_cat + torch.mul(feature_cat,self.sigmoid(feature_semantic)))


		ins_r = self.semantic_output(feature_semantic)

		result = self.cls_conv(feature_final)
		result = self.upsample4(result)

		if mode=='train':
			return result,edge_r,edge1_r,edge2_r,ins_r
		else:
			return result
