import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from ProteinPairsGenerator.nn import EdgeConvBatch, EdgeConvMod


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def get_graph_conv_layer(input_size, hidden_size, output_size):
    mlp = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    print("In : ", input_size)
    gnn = EdgeConvMod(nn=mlp, aggr="add")
    graph_conv = EdgeConvBatch(gnn, output_size, batch_norm=True, dropout=0.2)
    return graph_conv


class Net(nn.Module):
    def __init__(self, x_input_size, adj_input_size, x_feat_size, hidden_size, output_size):
        super().__init__()

        self.embed_x = nn.Sequential(
            nn.Embedding(x_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.embed_adj = (
            nn.Sequential(
                nn.Linear(adj_input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            if adj_input_size
            else None
        )
        self.embed_feat = (
            nn.Sequential(
                nn.Linear(x_feat_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            if x_feat_size
            else None
        )
        self.graph_conv_0 = get_graph_conv_layer((2 + bool(adj_input_size) + 2 * bool(x_feat_size)) * hidden_size, 2 * hidden_size, hidden_size)

        N = 3
        graph_conv = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv = _get_clones(graph_conv, N)

        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_attr, x_feat = None):

        edge_attr = (edge_attr + edge_attr_out) if edge_attr is not None else edge_attr_out

        x = self.embed_x(x)
        if x_feat is not None:
            x_feat = self.embed_feat(x_feat)
            x_out, edge_attr_out = self.graph_conv_0(torch.cat([x, x_feat], dim=-1), edge_index, edge_attr)
        else:
            x_out, edge_attr_out = self.graph_conv_0(x, edge_index, edge_attr)

        edge_attr = self.embed_adj(edge_attr) if edge_attr is not None else None

        x = x + x_out

        for i in range(3):
            x = F.relu(x)
            edge_attr = F.relu(edge_attr)
            x_out, edge_attr_out = self.graph_conv[i](x, edge_index, edge_attr)
            x = x + x_out
            edge_attr = edge_attr + edge_attr_out

        x = self.linear_out(x)

        return x
