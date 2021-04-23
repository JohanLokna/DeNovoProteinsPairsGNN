from typing import Tuple

import torch

from ProteinPairsGenerator.utils.amino_acids import AMINO_ACID_NULL, AMINO_ACIDS_MAP


def maskBERTOnehot(
    inSeq : torch.Tensor,
    substitutionMatrix : torch.Tensor,
    maskFrac : float = 0.15,
    keepFrac: float = 0.1,
    substituteFrac : float = 0.1,
    nullToken : int = AMINO_ACIDS_MAP[AMINO_ACID_NULL]
) -> Tuple[torch.Tensor, torch.BoolTensor]:

    """
    @param  inSeq: Sequence to be masked according to BERT training
    @type   inSeq: torch.Tensor [N, M]
    @param  substitutionMatrix: Symmetric matrix with subsitution probabilities
    @type   substitutionMatrix: torch.Tensor [M, M]
    @param  maskFrac: Fraction of elements in inSeq which will be maksed
    @type   maskFrac: float
    @param  keepFrac: Fraction of masked elements which will be kept unchanged
    @type   keepFrac: float
    @param  substituteFrac: Fraction of masked elements which will be swapped
    @type   substituteFrac: float
    @param  nullToken: Token to 
    @type   subsituteFrac: int
    @return: Return masked sequence and indicator of masked elements
    @rtype: torch.Tensor [N, M], torch.BoolTensor [N]
    @inv: 0 < maskFrac and maskFrac < 1
    @inv: 0 < keepFrac and 0 < subsituteFrac and keepFrac + subsituteFrac < 1
    @inv: nullToken < M
    """

    # Chose elements to mask
    n = inSeq.shape[0]
    nMask = int(maskFrac * n)
    idx = torch.multinomial(torch.ones(n), nMask, replacement=False).to(inSeq).to(dtype=torch.long)

    # Create mask
    mask = torch.empty(n).to(inSeq).to(dtype=torch.bool)
    mask.zero_()
    mask.scatter_(0, idx, True)

    # Determine indecies for different masks
    beginChange = int(keepFrac * nMask)
    sizeSub = int(substituteFrac * nMask)
    changeIdx = idx[beginChange:]
    subIdx = idx[beginChange:beginChange + sizeSub]
    nullIdx = idx[beginChange + sizeSub:]

    # Create masked sequence
    maskedSeq = inSeq.detach().clone()
    maskedSeq[changeIdx, :] = 0.0

    # Mask null token
    maskedSeq[nullIdx, nullToken] = 1.0

    # Mask substituted tokens
    probs = torch.matmul(inSeq[subIdx], substitutionMatrix)
    newTokens = torch.multinomial(probs, 1, replacement=True)
    onehotNewTokens = torch.empty_like(inSeq[subIdx])
    onehotNewTokens.zero_()
    onehotNewTokens.scatter_(1, newTokens, 1.0)
    maskedSeq[subIdx] = onehotNewTokens
    
    return maskedSeq, mask


def maskBERT(
    inSeq : torch.Tensor,
    substitutionMatrix : torch.Tensor,
    maskFrac : float = 0.15,
    keepFrac: float = 0.1,
    substituteFrac : float = 0.1,
    nullToken : int = AMINO_ACIDS_MAP[AMINO_ACID_NULL]
) -> Tuple[torch.Tensor, torch.BoolTensor]:

    """
    @param  inSeq: Sequence to be masked according to BERT training
    @type   inSeq: torch.Tensor [N]
    @param  substitutionMatrix: Symmetric matrix with subsitution probabilities
    @type   substitutionMatrix: torch.Tensor [M, M]
    @param  maskFrac: Fraction of elements in inSeq which will be maksed
    @type   maskFrac: float
    @param  keepFrac: Fraction of masked elements which will be kept unchanged
    @type   keepFrac: float
    @param  substituteFrac: Fraction of masked elements which will be swapped
    @type   substituteFrac: float
    @param  nullToken: Token to 
    @type   subsituteFrac: int
    @return: Return masked sequence and indicator of masked elements
    @rtype: torch.Tensor [N], torch.BoolTensor [N]
    @inv: 0 < maskFrac and maskFrac < 1
    @inv: 0 < keepFrac and 0 < subsituteFrac and keepFrac + subsituteFrac < 1
    @inv: nullToken < M
    @inv: (inSeq < M).all()
    """

    # Chose elements to mask
    n = inSeq.shape[0]
    nMask = int(maskFrac * n)
    idx = torch.multinomial(torch.ones(n), nMask, replacement=False).to(inSeq).to(dtype=torch.long)

    # Create mask
    mask = torch.empty(n).to(inSeq).to(dtype=torch.bool)
    mask.zero_()
    mask.scatter_(0, idx, True)

    # Determine indecies for different masks
    beginChange = int(keepFrac * nMask)
    sizeSub = int(substituteFrac * nMask)
    subIdx = idx[beginChange:beginChange + sizeSub]
    nullIdx = idx[beginChange + sizeSub:]

    # Create masked sequence
    maskedSeq = inSeq.detach().clone()

    # Mask null token
    maskedSeq[nullIdx] = nullToken

    # Mask substituted tokens
    probs = substitutionMatrix[inSeq[subIdx]].squeeze(1)
    newTokens = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
    maskedSeq[subIdx] = newTokens
    
    return maskedSeq, mask
