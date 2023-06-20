#!/usr/bin/env python
# encoding: utf-8
# Created by BIT09 at 2023/4/27
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GNNExplainer
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool


# Build the model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x


# Creating the function to train the model
def Train(data_loader, loss_func):
    model.train()
    # Iterate in batches over the training dataset
    for data in data_loader:
        # Perform a single forward pass
        out = model(data.x, data.edge_index, data.batch)
        # Compute the loss
        loss = loss_func(out, data.y)
        # Derive gradients
        loss.backward()
        # Update parameters based on gradients
        optimizer.step()
        # Clear gradients
        optimizer.zero_grad()


# function to test the model
def Test(data_loader):
    model.eval()
    correct = 0
    # Iterate in batches over the training/test dataset
    for data in data_loader:
        out = model(data.x, data.edge_index, data.batch)
        # Use the class with highest probability.
        pred = out.argmax(dim=1)
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return correct / len(test_loader.dataset)


if __name__ == '__main__':
    # Load the dataset
    dataset = TUDataset(root='../../data_root/TUDataset', name='MUTAG')
    # print details about the graph
    print(f'Dataset: {dataset}:')
    print("Number of Graphs: ", len(dataset))
    print("Number of Freatures: ", dataset.num_features)
    print("Number of Classes: ", dataset.num_classes)

    data = dataset[0]
    print(data)
    print("No. of nodes: ", data.num_nodes)
    print("No. of Edges: ", data.num_edges)
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    # Create train and test dataset
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:50]
    test_dataset = dataset[50:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs:{len(test_dataset)}')

    '''graphs in graph classification datasets are usually small,
    a good idea is to batch the graphs before inputting
    them into a Graph Neural Network to guarantee full GPU utilization__
    _In pytorch Geometric adjacency matrices are stacked in a diagonal fashion
    (creating a giant graph that holds multiple isolated subgraphs), a
    nd node and target features are simply concatenated in the node dimension:
    '''
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}')
        print('==============')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    # Build the model
    model = GNN(hidden_channels=64)
    print(model)

    # set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    # set the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model for 150 epochs
    for epoch in range(1, 100):
        Train(train_loader, loss_func=criterion)
        train_acc = Test(train_loader)
        test_acc = Test(test_loader)
        if epoch % 10 == 0:
            '''print(f'Epoch {epoch:>3} | Train Loss: {total_loss/len(train_loader):.3f} '
                  f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                  f'{val_loss/len(train_loader):.2f} | Val Acc: '
                  f'{val_acc/len(train_loader)*100:.2f}%')
            '''
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Explain the graph
    explainer = GNNExplainer(model, epochs=100, return_type='log_prob')
    data = dataset[0]
    node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)
    ax, G = explainer.visualize_subgraph(-1, data.edge_index, edge_mask, data.y)
    plt.show()
