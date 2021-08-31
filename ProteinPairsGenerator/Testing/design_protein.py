import torch
from ProteinPairsGenerator.utils import AMINO_ACID_NULL, AMINO_ACIDS_MAP, AMINO_ACIDS_BASE

"""
    Function for recursive protein design
"""
@torch.no_grad()
def designProtein(net : torch.nn.Module, kw_seq : str, in_seq, unsqueeze : bool, return_confidence : bool = False, **kwargs):
    
    # Can only design a single protein at the time
    assert len(in_seq.shape) == 1

    # Mask initially
    mask = in_seq != AMINO_ACIDS_MAP[AMINO_ACID_NULL]

    if return_confidence:
        confidence = torch.empty(in_seq.shape[0], len(AMINO_ACIDS_BASE), device=in_seq.device, dtype=torch.float)

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

        if return_confidence:
            confidence[max_residue] = output[max_residue, :]
    
    if return_confidence:
        return in_seq, confidence
    else:
        return in_seq

"""
    Function for recursive protein design combining two different models; one for the first part, the other for the last part
"""
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

