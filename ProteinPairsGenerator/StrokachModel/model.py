# General imports
import copy

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

# Local imports
from ProteinPairsGenerator.StrokachModel.edge_conv_mod import EdgeConvBatch, EdgeConvMod
from ProteinPairsGenerator.BERTModel import BERTModel


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def get_graph_conv_layer(input_size, hidden_size, output_size):
    mlp = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    gnn = EdgeConvMod(nn=mlp, aggr="add")
    graph_conv = EdgeConvBatch(gnn, output_size, batch_norm=True, dropout=0.2)
    return graph_conv


class Net(BERTModel):
    def __init__(self, x_input_size, adj_input_size, hidden_size, output_size, N = 3):
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
        self.graph_conv_0 = get_graph_conv_layer((2 + bool(adj_input_size)) * hidden_size, 2 * hidden_size, hidden_size)

        graph_conv = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv = _get_clones(graph_conv, N)

        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_attr):

        edge_attr = self.embed_adj(edge_attr) if edge_attr is not None else None

        x = self.embed_x(x)
        x_out, edge_attr_out = self.graph_conv_0(x, edge_index, edge_attr)

        x = x + x_out
        edge_attr = (edge_attr + edge_attr_out) if edge_attr is not None else edge_attr_out

        for graph_layer in self.graph_conv:
            x = F.relu(x)
            edge_attr = F.relu(edge_attr)
            x_out, edge_attr_out = graph_layer(x, edge_index, edge_attr)
            x = x + x_out
            edge_attr = edge_attr + edge_attr_out

        x = self.linear_out(x)

        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'valLoss'
       }
    
    def step(self, batch):
        
        loss = 0
        nCorrect = 0
        nTotal = 0
        for x in batch:

            output = self(x.seq, x.edge_index, x.edge_attr)[x.mask]
            yTrue = x.y[x.mask]

            loss += self.criterion(output, yTrue)

            yPred = torch.argmax(output.data, 1)
            nCorrect += (yPred == yTrue).sum()
            nTotal += torch.numel(yPred)

        return {
            "loss" : loss,
            "nCorrect" : nCorrect,
            "nTotal" : nTotal
        }
