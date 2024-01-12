import torch.distributed as dist
import os
import math
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np


class CFDistributedSampler(DistributedSampler):
    """
    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas = None,
                 rank = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, 
                 per_gpu_batch_size: int=None, closek_info = None, fark_info=None, break_near_total=False,
                 pick_nck = False) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True
        self.total_size = len(self.dataset)
        self.shuffle = shuffle
        self.seed = seed
        self.per_gpu_batch_size = per_gpu_batch_size #make sure it is something like 65, 129, 257 and so on as we pick 1 + [close] + [far]
        self.is_kappa_close = closek_info != None
        self.is_kappa_far = fark_info != None
        self.closek_info = np.load(closek_info) if closek_info else None #TODO: get topk info for each 
        self.fark_info = np.load(fark_info) if fark_info else None
        self.length = self.total_size
        self.break_near_total = break_near_total
        self.pick_nck = pick_nck  #Doesnt do anything

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        assert len(indices) == self.total_size
        #per gpu batch size
        per_gpu_batch_size = self.per_gpu_batch_size
        if self.is_kappa_close:
          assert self.closek_info.shape[0] == len(indices), f'{self.closek_info.shape[0]}!={len(indices)} closek != indices'
          if self.is_kappa_far:
            assert self.fark_info.shape[0] == len(indices), f'{self.fark_info.shape[0]}!={len(indices)} fark != indices'
            kappa_close = per_gpu_batch_size // 2
            kappa_far = per_gpu_batch_size // 2
          else:
            kappa_close = (per_gpu_batch_size // 2) * 2
            kappa_far = 0
        else:
          if self.is_kappa_far:
            assert self.fark_info.shape[0] == len(indices), f'{self.fark_info.shape[0]}!={len(indices)} fark != indices'
            kappa_far = (per_gpu_batch_size // 2) * 2
            kappa_close = 0
          else:
            kappa_far = 0
            kappa_close = 0

        def _get_k(current_index, knn_info, kappa):
          g2 = torch.Generator()
          g2.manual_seed(current_index)
          if kappa > 0:
              return knn_info[indices[current_index],:kappa]
          return None
        
        def _get_next_index(current_index):
            while indices[current_index] in bag_of_indices and current_index < len(indices):
              current_index += 1
            return current_index

        if self.is_kappa_close or self.is_kappa_far:
          bag_of_indices = set()
          current_rank = 0
          rankwise_indices = {i:[] for i in range(self.num_replicas)}
          start_index = 0
          while len(bag_of_indices) <= self.total_size-per_gpu_batch_size: #Final breaking condition. Produces more batches due to duplication of nearby samples
            index = []
            start_index = _get_next_index(start_index)
            index.append(indices[start_index])
            bag_of_indices.add(indices[start_index])
            # GET CLOSE AND FAR K
            print(len(bag_of_indices), 'current bag size! and total length is ', len(rankwise_indices[current_rank]))
            closest_k_indices = _get_k(start_index, self.closek_info, kappa_close)
            if kappa_close > 0:
              bag_of_indices.update(closest_k_indices)
              index.extend(closest_k_indices)
            farthest_k_indices = _get_k(start_index, self.fark_info, kappa_far)
            if kappa_far > 0:
              bag_of_indices.update(farthest_k_indices)
              index.extend(farthest_k_indices)
            rankwise_indices[current_rank].extend(index)
            current_rank = (current_rank + 1)%self.num_replicas
            if self.break_near_total and len(rankwise_indices[current_rank]) >= len(indices):
                break
          exit() 
          #Check if all GPUs got equal number of indices
          dists = [len(rankwise_indices[i]) for i in range(self.num_replicas)]
          mx = max(dists)
          mn = min(dists)
          if mx != mn:
            assert mx-mn == kappa_far+kappa_close+1, f'difference between max and min number of samples per rank is {mx-mn}'
            for k, v in rankwise_indices.items():
              if len(v) != mx:
                start_index = (start_index + 1)%len(indices)
                rankwise_indices[k].append(indices[start_index])
                closest_k_indices = _get_k(start_index, self.closek_info, kappa_close)
                if kappa_close > 0:
                  rankwise_indices[k].extend(closest_k_indices)
                farthest_k_indices = _get_k(start_index, self.fark_info, kappa_far)
                if kappa_far > 0:
                  rankwise_indices[k].extend(farthest_k_indices)
                assert len(rankwise_indices[k]) == mx, 'After adding more samples the legnth is still not equal! {len(rankwise_indices[k])} != {mx}'
          self.length = mx
          if self.rank == 0:
              print(f'length is {mx}. bag_len = {len(bag_of_indices)}. Index is at {start_index}')
          return iter(rankwise_indices[self.rank])
        else:
          raise NotImplementedError('Far or Close both unset. Use train coss instead for no sampling!')

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
