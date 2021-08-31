# Pytorch imports
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

"""
    Base class dataloader which is specialized for the different models
"""
class BERTLoader(DataLoader):

    def __init__(
        self,
        dataset : Subset,
        *args,
        **kwargs
    ) -> None:

        if ("sampler" in kwargs) and isinstance(kwargs["sampler"], DistributedSampler):
            rank = kwargs["sampler"].rank
            size = kwargs["sampler"].num_replicas

            # Assign datapoints according to set index of first element of batch
            newIndecies = [x for x in dataset.indices if (isinstance(x, tuple) and x[1] % size == rank) \
                                                      or (isinstance(x, list) and isinstance(x[0], tuple) and x[0][1] % size == rank)]
            dataset = Subset(dataset=dataset.dataset, indices=newIndecies)
            kwargs["sampler"] = None
        kwargs["shuffle"] = False

        super().__init__(
            dataset=dataset,
            *args,
            **kwargs
        )
