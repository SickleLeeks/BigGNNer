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
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True, node=False):  # disable node_feat_mask by default
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.node = node

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if self.node:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.edge_mask = torch.nn.Parameter(torch.zeros(E) * 50)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        if self.node:
            self.node_feat_masks = None
        self.edge_mask = None

    def __num_hops__(self):
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
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
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow=self.__flow__())
            x = x[subset]
        else:
            x = x
            edge_index = edge_index
            row, col = edge_index
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            edge_mask[:] = True
            mapping = None

        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def __graph_loss__(self, log_logits, pred_label):
        loss = -torch.log(log_logits[0, pred_label])
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs):
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx is not None:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            edge_mask = edge_mask[hard_edge_mask]
            subset = subset.tolist()
            if y is None:
                y = torch.zeros(edge_index.max().item() + 1,
                                device=edge_index.device)
            else:
                y = y[subset].to(torch.float) / y.max().item()
                y = y.tolist()
        else:
            subset = []
            for index, mask in enumerate(edge_mask):
                node_a = edge_index[0, index]
                node_b = edge_index[1, index]
                if node_a not in subset:
                    subset.append(node_a.item())
                if node_b not in subset:
                    subset.append(node_b.item())
            y = [y for i in range(len(subset))]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=len(y)).to('cpu')
        G = to_networkx(data, edge_attrs=['att'])  # , node_attrs=['y']
        mapping = {k: i for k, i in enumerate(subset)}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 800
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, node_color=y, **kwargs)
        nx.draw_networkx_labels(G, pos, **kwargs)

        return ax, G

    def explain_graph(self, data, **kwargs):
        self.model.eval()
        self.__clear_masks__()
        x, edge_index, batch = data.x, data.edge_index, data.batch

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, _, hard_edge_mask, kwargs = self.__subgraph__(node_idx=None, x=x, edge_index=edge_index,
                                                                     **kwargs)
        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(data, **kwargs)
            probs_Y = torch.softmax(log_logits, 1)
            pred_label = probs_Y.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        if self.node:
            optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                         lr=self.lr)
        else:
            optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        epoch_losses = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            optimizer.zero_grad()
            if self.node:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()

            log_logits = self.model(data, **kwargs)
            pred = torch.softmax(log_logits, 1)
            loss = self.__graph_loss__(pred, pred_label)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_losses.append(epoch_loss)

        edge_mask = self.edge_mask.detach().sigmoid()
        print(edge_mask)

        self.__clear_masks__()

        return edge_mask, epoch_losses

    def __repr__(self):
        return f'{self.__class__.__name__}()'
