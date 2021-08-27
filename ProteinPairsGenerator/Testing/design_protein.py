import torch
from ProteinPairsGenerator.utils import AMINO_ACID_NULL, AMINO_ACIDS_MAP

@torch.no_grad()
def designProtein(net : torch.nn.Module, kw_x : str, dim : int, **kwargs):
    
    # Can only design a single protein at the time
    assert len(kwargs[kw_x].shape) == 1

    # Mask initially
    mask = kwargs[kw_x] == AMINO_ACIDS_MAP[AMINO_ACID_NULL]

    while not mask.any():

        # Predict based on current predictions
        output = net(**kwargs)

        # Normalize predictions
        output = torch.softmax(output, dim=dim)

        # Get per residue max prediction
        # Set already predictions to by -1 in order not to change them again 
        max_pred, max_index = output.max(dim=dim)
        max_pred[~mask] = -1
        _, max_residue = max_pred.max(dim=0)

        # Update with max prediction
        kwargs[kw_x][max_residue] = max_index[max_residue]
        mask[max_residue] = False
    
    return kwargs[kw_x]
