import torch
import torch.nn as nn
from net.BSANet import BSANet
#from net.BANet import BANet

def generate_net(cfg):
	if cfg.MODEL_NAME == 'BSANet' or cfg.MODEL_NAME == 'BSANet+':
		return BSANet(cfg)
	if cfg.MODEL_NAME == 'FCN' or cfg.MODEL_NAME == 'FCN+':
		return BSANet(cfg)
	if cfg.MODEL_NAME == 'BANet' or cfg.MODEL_NAME == 'BANet+':
		pass
		return BANet(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
