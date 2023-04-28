#!/usr/bin/env python
# encoding: utf-8
# Created by BIT09 at 2023/4/28
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    r"""
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """
    coeffs = {
        'edge_size': 0.001,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1
    }

    # disable node_feat_mask by default
    def __init__(self, model, epochs=100, lr=0.01, log=True, node=False):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.node = node

    def __set_masks__(self, x, edge_index, init='normal'):
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        if self.node:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.edge_mask = torch.nn.Parameter(torch.zeros(E) * 50)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__=True
                module.__edge_mask__=self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        if self.node:
            self.node_feat_mask = None
        self.edge_mask = None

    def __num_hops__(self):
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops+=1
        return num_hops

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        if node_idx is not None:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True, num_nodes=num_nodes, flow=self.__flow__()
            )
            x = x[subset]
        else:
            x = x
            edge_index = edge_index
            row, col = edge_index
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
