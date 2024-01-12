import torch.distributed as dist
import os
import math
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np


class CFBatchSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas = None,
                 rank = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True,
                 batch_size=None, closek_info=None, nearest_k=None, total_k=None):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size // self.num_replicas 
        self.closek_info = np.load(closek_info) if closek_info else None
        self.nearest_k = nearest_k
        self.total_k = total_k

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        indices = indices[self.rank:self.total_size:self.num_replicas]

        def _get_k(current_index):
            if self.total_k != self.nearest_k:
                return np.random.choice(self.closek_info[current_index][:self.total_k], self.nearest_k, replace=False)
            else:
                return self.closek_info[current_index][:self.nearest_k]
        
        current_batch, batches = [], []
        current_length = 0
        for sample in indices:
            current_batch.append(sample)
            current_batch.extend(_get_k(sample))
            current_length += 1
            if current_length == self.batch_size:
                batches.append(current_batch)
                current_batch = []
                current_length = 0

        return iter(batches)
        
    def __len__(self) -> int:
        return len(self.dataset) // (self.batch_size*self.num_replicas)

class ClusterCFSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas = None,
                 rank = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True,
                 batches_info=None):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.epoch = 0
        self.seed = seed
        self.batches_info = np.load(batches_info, allow_pickle=True).item()
        self.length = len(self.batches_info[self.epoch][self.rank])

    def __iter__(self):
        self.length = len(self.batches_info[self.epoch][self.rank])
        return iter(self.batches_info[self.epoch][self.rank])
    
    def __len__(self) -> int:
        return self.length
        

class ClusterBatchSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas = None,
                 rank = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True,
                 batches_info=None):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.epoch = 0
        self.seed = seed
        self.batches_info = torch.load(batches_info)['batches']

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.batches_info), generator=g).tolist()  # type: ignore[arg-type]
        indices = indices[self.rank:self.total_size:self.num_replicas]

        for index in indices:
            yield self.batches_info[index]

        
    def __len__(self) -> int:
        return len(self.batches_info) // self.num_replicas
