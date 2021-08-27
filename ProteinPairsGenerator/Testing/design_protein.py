import torch
from ProteinPairsGenerator.utils import AMINO_ACID_NULL, AMINO_ACIDS_MAP

@torch.no_grad()
def designProtein(net : torch.nn.Module, kw_seq : str, in_seq, unsqueeze : bool, **kwargs):
    
    # Can only design a single protein at the time
    assert len(in_seq.shape) == 1

    # Mask initially
    mask = in_seq != AMINO_ACIDS_MAP[AMINO_ACID_NULL]

    while not mask.all():

        # Predict based on current predictions
        kwargs[kw_seq] = in_seq.unsqueeze(0) if unsqueeze else in_seq
        output = net(**kwargs)

        if unsqueeze:
            output = output.squeeze(0)

        # Normalize predictions
        output = torch.softmax(output, dim=-1)

        # Get per residue max prediction
        # Set already predictions to by -1 in order not to change them again 
        max_pred, max_index = output.max(dim=-1)
        max_pred[mask] = -1
        _, max_residue = max_pred.max(dim=0)

        # Update with max prediction
        in_seq[max_residue] = max_index[max_residue]
        mask[max_residue] = True
    
    return in_seq

@torch.no_grad()
def designProteinHybrid(net1 : torch.nn.Module, net2 : torch.nn.Module, alpha : float,
                        kw_seq : str, in_seq, unsqueeze : bool, **kwargs):
    
    # Can only design a single protein at the time
    assert len(in_seq.shape) == 1

    # Mask initially
    mask = in_seq != AMINO_ACIDS_MAP[AMINO_ACID_NULL]
    
    k = alpha * torch.numel(in_seq)
    i = 0
    while not mask.all():

        # Predict based on current predictions
        kwargs[kw_seq] = in_seq.unsqueeze(0) if unsqueeze else in_seq
        output = (net1 if i < k else net2)(**kwargs)

        if unsqueeze:
            output = output.squeeze(0)

        # Normalize predictions
        output = torch.softmax(output, dim=-1)

        # Get per residue max prediction
        # Set already predictions to by -1 in order not to change them again 
        max_pred, max_index = output.max(dim=-1)
        max_pred[mask] = -1
        _, max_residue = max_pred.max(dim=0)

        # Update with max prediction
        in_seq[max_residue] = max_index[max_residue]
        mask[max_residue] = True
        i += 1
    
    return in_seq

