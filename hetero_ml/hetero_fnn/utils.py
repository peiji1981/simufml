from typing import List

import torch as th
from torch import nn
from torch.nn import functional as F
import math

class PartialEmbedding(nn.Module):
    def __init__(
        self, 
        feature_size, 
        embed_size, 
        total_fs
    ):
        super().__init__()
        self.total_fs = total_fs
        self.em = nn.Embedding(feature_size, embed_size)
    
    def forward(self, x):
        bs = x.shape[0]
        vx = th.mul(x[:,:,None], self.em.weight[None,:,:])
        return vx.reshape(bs,-1)

    def initialize(self):
        nn.init.normal_(self.em.weight, 0, std=1/math.sqrt(self.total_fs))


class MLP(nn.Module):
    def __init__(
        self,
        arch: List[int]
    ):
        """
        Arguments:
            arch:
                [input_size, neurons_of_layer0, neurons_of_layer1, ...]
        """
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(arch[i], arch[i+1]) for i in range(len(arch)-1)])

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        return self.fcs[-1](x)

    def initialize(self):
        for fc in self.fcs:
            fc.reset_parameters()