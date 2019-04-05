import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        
        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        
        # weight bases
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        
        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
        
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
