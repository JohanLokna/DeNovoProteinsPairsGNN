from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import copy

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from ProteinPairsGenerator.IngrahamModel.self_attention import *
from ProteinPairsGenerator.IngrahamModel.protein_features import ProteinFeatures
from ProteinPairsGenerator.BERTModel import BERTModel
from ProteinPairsGenerator.IngrahamModel.noam_opt import NoamOpt


def loss_smoothed(S, log_probs, mask, vocab_size, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, num_classes = vocab_size).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


class Struct2Seq(BERTModel):
    def __init__(
        self, 
        in_size, 
        node_features, 
        edge_features,
        hidden_dim, 
        num_encoder_layers=3, 
        num_decoder_layers=3,
        out_size=20, 
        k_neighbors=30,
        protein_features='full',
        augment_eps=0.,
        dropout=0.1, 
        forward_attention_decoder=True, 
        use_mpnn=False
    ) -> None:
        """ Graph labeling network """
        super(Struct2Seq, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.in_size = in_size
        self.out_size = out_size
        self.k_neighbors = k_neighbors

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, 
            edge_features, 
            top_k=k_neighbors,
            features_type=protein_features, 
            augment_eps=augment_eps,
            dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(self.in_size, hidden_dim)
        layer = TransformerLayer if not use_mpnn else MPNNLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.forward_attention_decoder = forward_attention_decoder
        self.decoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, self.out_size, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _autoregressive_mask(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, device=E_idx.device)
        ii = ii.view((1, -1, 1))
        mask = E_idx - ii < 0
        mask = mask.type(torch.float32)

        # Debug 
        # mask_scatter = torch.zeros(E_idx.shape[0],E_idx.shape[1],E_idx.shape[1]).scatter(-1, E_idx, mask)
        # mask_reduce = gather_edges(mask_scatter.unsqueeze(-1), E_idx).squeeze(-1)
        # plt.imshow(mask_reduce.data.numpy()[0,:,:])
        # plt.show()
        # plt.imshow(mask.data.numpy()[0,:,:])
        # plt.show()
        return mask

    def forward(self, X, S, L, mask):
        """ Graph-conditioned sequence model """

        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        # Decoder uses masked self-attention
        mask_attend = self._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        
        if self.forward_attention_decoder:
            mask_fw = mask_1D * (1. - mask_attend)
            h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_ESV, mask_V=mask)

        logits = self.W_out(h_V) 
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def step(self, batch):

        (X, S, l, v), y, mask = batch
        output = self(X, S, l, v)
        
        _, loss = loss_smoothed(y, output, mask, self.out_size)

        yPred = torch.argmax(output.data, 2)
        nCorrect = ((yPred == y) * mask).sum()
        nTotal = torch.sum(mask)

        return {
            "loss" : loss,
            "nCorrect" : nCorrect,
            "nTotal" : nTotal
        }

    # def configure_optimizers(self):
    #     optimizer = NoamOpt.getStandard(self.parameters(), self.hidden_dim)
    #     return {
    #        'optimizer': optimizer,
    #        'monitor': 'valLoss'
    #     }
