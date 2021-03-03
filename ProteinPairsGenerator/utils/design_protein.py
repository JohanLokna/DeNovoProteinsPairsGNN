import torch

@torch.no_grad()
def design_protein(net, x, edge_index, edge_attr):
    mask = (x == 20)
    if not mask.any():
        return x
    output = net(x, edge_index, edge_attr)
    output = torch.softmax(output, dim=1)
    max_pred, max_index = output.max(dim=1)
    max_pred[~mask] = -1
    _, max_residue = max_pred.max(dim=0)
    x[max_residue] = max_index[max_residue]
    return design_protein(net, x, edge_index, edge_attr)
