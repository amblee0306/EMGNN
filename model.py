import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCNv2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, layer, activation, act_before_dropout=False):
        super(GCNv2, self).__init__()

        self.dropout = dropout
        self.num_layer = layer
        self.activation = activation
        self.layers = nn.ModuleList()
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, nfeat))
        else:
            self.layers.append(GraphConvolution(nfeat, nhid, nfeat))
            for i in range(self.num_layer-2):
                self.layers.append(GraphConvolution(nhid, nhid, nfeat))
            self.layers.append(GraphConvolution(nhid, nclass, nfeat))
        self.act_before_dropout = act_before_dropout

    def forward(self, adj, x):
        h = x.clone().detach()
        for i, layer in enumerate(self.layers):
            if i == self.num_layer - 1:
                # last layer without activation or dropout
                x = layer(x, adj, h)
            else:
                if self.act_before_dropout:
                    x = self.activation(layer(x, adj, h))
                    x = F.dropout(x, self.dropout, training=self.training)
                else:
                    x = F.dropout(x, self.dropout, training=self.training)
                    x = self.activation(layer(x, adj, h))
        return x
