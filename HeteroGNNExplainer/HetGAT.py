#!/usr/bin/env python
# encoding: utf-8
# Created by BIT09 at 2023/5/10
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
                ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'rev_writes', 'author'): GATConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['author'])


if __name__ == '__main__':
    dataset = OGB_MAG(root='../../data_root/OGB_MAG', preprocess='metapath2vec', transform=T.ToUndirected())
    data = dataset[0]

    model = HeteroGNN(hidden_channels=64, out_channels=dataset.num_classes, num_layers=2)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)