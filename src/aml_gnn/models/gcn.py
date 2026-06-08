import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCNNodeClassifier(nn.Module):
    """
    GCN para clasificación binaria de nodos.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)

        return out
