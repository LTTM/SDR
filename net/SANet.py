import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d


class SANet(nn.Module):

    def __init__(self):
        super(SANet, self).__init__()

        self.softmax =nn.Softmax(dim=-1)
        self.gamma =nn.Parameter(torch.zeros(1))

    def forward(self, x,feature_s):
        [b, c, row, col] = feature_s.size()

        project_q=feature_s.view(b,c,-1)
        project_k=feature_s.view(b,c,-1).permute(0,2,1)

        energy=torch.bmm(project_q,project_k)
        energy_new=torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy

        attention=self.softmax(energy_new)

        project_value=x.view(b,c,-1)
        result=torch.bmm(attention,project_value)

        result=result.view(b,c,row,col)

        result=result*self.gamma+x
        return result

