import torch


def normalize_cart_distances(cart_distances):
    return (cart_distances - 6) / 12


def normalize_seq_distances(seq_distances):
    return (seq_distances - 0) / 68.1319


def transform_edge_attr(data):
    cart_distances = data.edge_attr
    cart_distances = normalize_cart_distances(cart_distances)
    seq_distances = (data.edge_index[1] - data.edge_index[0]).to(torch.float).unsqueeze(1)
    seq_distances = normalize_seq_distances(seq_distances)
    data.edge_attr = torch.cat([cart_distances, seq_distances], dim=1)
    return data
