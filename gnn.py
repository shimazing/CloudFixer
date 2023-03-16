from hesiod import hcfg
import torch
import wandb
import glob
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn


class GCN(torch.nn.Module):
    def __init__(self, num_features=1024, num_classes=10):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features)
        self.conv3 = GCNConv(num_features, num_classes)
        self.linear1 = nn.Linear(10, num_features)

    def forward(self, x, pseudo_y, edge_index, egde_values=None):

        y = self.linear1(pseudo_y)
        x = self.conv1(x+y, edge_index, egde_values)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        f = self.conv2(x, edge_index, egde_values)
        x = f.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index, egde_values)
        return f, x

