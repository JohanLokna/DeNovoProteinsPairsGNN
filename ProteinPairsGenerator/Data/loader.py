# Pytorch imports
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


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
            newIndecies = [x for x in dataset.indices if (isinstance(x, tuple) and x[0] % size == rank) \
                                                      or (isinstance(x, list) and isinstance(x[0], tuple) and x[0][0] % size == rank)]
            dataset = Subset(dataset=dataset.dataset, indices=newIndecies)
            kwargs["sampler"] = None
        kwargs["shuffle"] = False

        super().__init__(
            dataset=dataset,
            *args,
            **kwargs
        )
