import argparse
import os
import math
import torch
import numpy as np

class BatchInfo:
    def __init__(self, num_replicas: int=1, per_gpu_batch_size: int=None, closek_info = None, fark_info=None, break_near_total=False, pick_nck = False):
        self.num_replicas = num_replicas 
        self.per_gpu_batch_size = per_gpu_batch_size #make sure it is something like 65, 129, 257 and so on as we pick 1 + [close] + [far]
        self.is_kappa_close = closek_info != None
        self.is_kappa_far = fark_info != None
        self.closek_info = np.load(closek_info) 
        self.fark_info = np.load(fark_info) if fark_info else None
        
        self.length = self.closek_info.shape[0]
        
        self.break_near_total = break_near_total
        self.pick_nck = pick_nck  #Doesnt do anything
        
        self.seed = 42

    def __call__(self, epoch):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        indices = torch.randperm(self.length, generator=g).tolist()  # type: ignore[arg-type]

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
          if kappa > 0:
              return knn_info[indices[current_index],:kappa]
          return None
        
        def _get_next_index(current_index):
            while indices[current_index] in bag_of_indices and current_index < len(indices):
              current_index += 1
            return current_index

        bag_of_indices = set()
        current_rank = 0
        rankwise_indices = {i:[] for i in range(self.num_replicas)}
        start_index = 0

        while len(bag_of_indices) <= self.length-per_gpu_batch_size: #Final breaking condition. Produces more batches due to duplication of nearby samples
            index = []
            start_index = _get_next_index(start_index)
            index.append(indices[start_index])
            bag_of_indices.add(indices[start_index])
            # GET CLOSE AND FAR K
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
        
        print(f'EPOCH {epoch}: per gpu batch is {mx}. UNIQUE BAG LEN = {len(bag_of_indices)}')
        return rankwise_indices

class ClusterBatchInfo(BatchInfo):
    def __init__(self, cluster_info=None, **kwargs):
        super().__init__(**kwargs)
        self.cluster_info = torch.load(cluster_info)['batches']

    def __call__(self, epoch):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        indices = torch.randperm(len(self.cluster_info), generator=g).tolist()  # type: ignore[arg-type]

        per_gpu_batch_size = self.per_gpu_batch_size
        
        if self.is_kappa_close:
          if self.is_kappa_far:
            kappa_close = per_gpu_batch_size // 2
            kappa_far = per_gpu_batch_size // 2
          else:
            kappa_close = (per_gpu_batch_size // 2) * 2
            kappa_far = 0
        else:
          if self.is_kappa_far:
            kappa_far = (per_gpu_batch_size // 2) * 2
            kappa_close = 0
          else:
            kappa_far = 0
            kappa_close = 0

        def _get_k(current_index, knn_info, kappa):
          if kappa > 0:
              return knn_info[current_index,:kappa]
          return None
        
        def _get_next_index(si, ci):
            while si < len(self.cluster_info[indices[ci]]) and self.cluster_info[indices[ci]][si] in bag_of_indices:
              si += 1
            return si

        def _find_good_cluster(si, ci):
            while True:
                while cluster_marker[ci] >= len(self.cluster_info[indices[ci]]):
                    ci = (ci + 1)%len(cluster_marker)
                si = cluster_marker[ci]
                si = _get_next_index(si, ci)
                if si < len(self.cluster_info[indices[ci]]):
                    break
                else:
                    cluster_marker[ci] = si
            cluster_marker[ci] = si
            return si, ci

        bag_of_indices = set()
        current_rank = 0
        rankwise_indices = {i:[] for i in range(self.num_replicas)}
        start_index = 0
        cluster_index = 0
        cluster_marker = [0 for _ in range(len(self.cluster_info))] #tracks where the index is within each cluster
        while len(bag_of_indices) <= self.length * 0.95: #Final breaking condition. Produces more batches due to duplication of nearby samples
            
            index = []
            # pick a non empty cluster next
            start_index, cluster_index = _find_good_cluster(cluster_marker[cluster_index], cluster_index) 
            
            #good cluster found
            value = self.cluster_info[indices[cluster_index]][start_index]
            bag_of_indices.add(value)
            index.append(value)

            # GET CLOSE AND FAR K
            if kappa_close > 0:
              closest_k_indices = _get_k(value, self.closek_info, kappa_close)
              bag_of_indices.update(closest_k_indices)
              index.extend(closest_k_indices)

            if kappa_far > 0:
              farthest_k_indices = _get_k(value, self.fark_info, kappa_far)
              bag_of_indices.update(farthest_k_indices)
              index.extend(farthest_k_indices)
            
            rankwise_indices[current_rank].extend(index)
            current_rank = (current_rank + 1)%self.num_replicas
            cluster_index = (cluster_index + 1)%len(indices)
            
            if self.break_near_total and len(rankwise_indices[current_rank]) >= len(indices)//self.num_replicas:
                break
        
        #Check if all GPUs got equal number of indices
        dists = [len(rankwise_indices[i]) for i in range(self.num_replicas)]
        mx = max(dists)
        mn = min(dists)
        if mx != mn:
            assert mx-mn == kappa_far+kappa_close+1, f'difference between max and min number of samples per rank is {mx-mn}'
            for k, v in rankwise_indices.items():
              if len(v) != mx:
                start_index, cluster_index = _find_good_cluster(cluster_marker[cluster_index], cluster_index) 
            
                #good cluster found
                value = self.cluster_info[indices[cluster_index]][start_index]
                bag_of_indices.add(value)
                index.append(value)
                rankwise_indices[k].append(value)
                if kappa_close > 0:
                  closest_k_indices = _get_k(value, self.closek_info, kappa_close)
                  rankwise_indices[k].extend(closest_k_indices)
                  bag_of_indices.update(closest_k_indices)
                if kappa_far > 0:
                  farthest_k_indices = _get_k(value, self.fark_info, kappa_far)
                  rankwise_indices[k].extend(farthest_k_indices)
                  bag_of_indices.update(farthest_k_indices)
                assert len(rankwise_indices[k]) == mx, 'After adding more samples the legnth is still not equal! {len(rankwise_indices[k])} != {mx}'
        
        print(f'EPOCH {epoch}: per gpu batch is {mx}. UNIQUE BAG LEN = {len(bag_of_indices)}')
        return rankwise_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Crete offline batches to be used in training!')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--num-gpus', required=True, type=int, default=None)
    parser.add_argument('--epochs', required=True, type=int, default=None)
    parser.add_argument('--per-gpu-batch-size', required=True, type=int, default=None)
    parser.add_argument('--closek-info', required=True, type=str, default=None)
    parser.add_argument('--cluster-info', required=True, type=str, default=None)
    parser.add_argument('--fark-info', required=False, type=str, default=None)
    
    args = parser.parse_args()
    filename = f'GPUS_{args.num_gpus}_EPOCHS_{args.epochs}_BSperGPU_{args.per_gpu_batch_size}_CLOSE_{args.closek_info is not None}_FAR_{args.fark_info is not None}_CLUSTER_{args.cluster_info is not None}.batches.npy'
    batches_info = {i: None for i in range(args.epochs)}
    
    if not args.cluster_info:
         batch_composer = BatchInfo(num_replicas=args.num_gpus, per_gpu_batch_size=args.per_gpu_batch_size, closek_info=args.closek_info, fark_info=args.fark_info) 
    else: 
         batch_composer = ClusterBatchInfo(cluster_info=args.cluster_info, num_replicas=args.num_gpus, per_gpu_batch_size=args.per_gpu_batch_size, closek_info=args.closek_info, fark_info=args.fark_info) 
    
    for epoch in range(args.epochs):
        batches = batch_composer(epoch)
        batches_info[epoch] = batches

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, filename), batches_info)
