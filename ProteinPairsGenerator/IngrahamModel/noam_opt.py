import torch
from torch import optim

class NoamOpt(optim.Adam):
    "Optim wrapper that implements rate."
    def __init__(self, hiddenSize, factor, warmup, **kwAdamParams):

        super().__init__(**kwAdamParams)

        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hiddenSize = hiddenSize
        self._rate = 0
        
    def step(self, *args, **kwargs):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in super().param_groups:
            p['lr'] = rate
        self._rate = rate
        super().step(*args, **kwargs)
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.hiddenSize ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        super().zero_grad()
    
    @staticmethod
    def getStandard(params, hiddenSize):
        return NoamOpt(hiddenSize, 2, 4000, params=params, lr=0, betas=(0.9, 0.98), eps=1e-9)
