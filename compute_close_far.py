# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
from collections import OrderedDict
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
#from torchvision import models as torchvision_models
import seed.models as models 
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import numpy as np

def get_dataloader(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(int(args.image_size*1.05), interpolation=3),
        pth_transforms.CenterCrop(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ImageFolder(root=os.path.join(args.data_path, 'train'), transform=transform)
    if args.subset:
        indices, counts  = [], {i:[] for i in range(1000)}
        for idx, target in enumerate(dataset.targets):
            counts[target].append(idx)
        for _, count in counts.items():
            subset_size = int(len(count)*args.subset)
            indices = indices + count[:subset_size]
        dataset = Subset(dataset, indices)
        print('Subset initialized with total size', len(dataset))
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    return transform, dataset, data_loader


def extract_feature_pipeline(model, args):
    transform, dataset_train, data_loader_train = get_dataloader(args)

    print(f"Extracting features for train set...")
    train_features = extract_features(model, data_loader_train)

    return train_features


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    features = []
    indices = []
    for i, (samples, _) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        feats = model(samples).cpu()
        feats = nn.functional.normalize(feats, dim=1, p=2)
        features.append(feats.cpu())
        if i%100==0:
            print('Round', i)
    print('Last Round', i)
    return torch.cat(features, dim=0)

@torch.no_grad()
def gen_nn_indices(train_features, k):
    closek_info, fark_info = [], []
    num_train_features = train_features.shape[0]
    print('training_shape', train_features.shape)
    for i in range(0, num_train_features, 256):
        features = train_features[i:min(i+256, num_train_features), :]
        similarity = torch.mm(features, train_features.T)
        _, close_ids = similarity.topk(k+1, largest=True, sorted=True)
        _, far_ids = similarity.topk(k+1, largest=False, sorted=True)
        if i == 0:
           assert (close_ids[:, 0].cpu().numpy() == [_ for _ in range(close_ids.shape[0])]).all()
        closek_info.append(close_ids[:, 1:k+1])      
        fark_info.append(far_ids[:, :k])
    closek_info = torch.cat(closek_info, dim=0).cpu().numpy()
    fark_info = torch.cat(fark_info, dim=0).cpu().numpy()
    print('matrix_shape', fark_info.shape)
    print('matrix_shape', closek_info.shape)
    return closek_info, fark_info


class ReturnIndexDataset():
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, idx):
        img, _ = self.dataset(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument("--checkpoint_key", default="state_dict", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--subset', default=None, type=float)
    args = parser.parse_args()
    
    if 'vit' not in args.arch:
        model = models.__dict__[args.arch](num_classes=0)
    else:
        model = models.__dict__[args.arch](num_classes=0, is_eval=True)
    
    try:
        model.fc = nn.Identity()
    except:
        model.classifier = nn.Identity()
    model.cuda()

    if args.pretrained_weights:
        ckpt = torch.load(args.pretrained_weights)
        msg = model.load_state_dict(ckpt[args.checkpoint_key], strict=False)
        if len(msg.missing_keys) > 2:
            nd = OrderedDict()
            for k, v in ckpt[args.checkpoint_key].items():
                if 'module.' in k: #NEED FOR DDP METHODS
                    nd[k[7:]] = v
                #if 'encoder.' in k:
                #    nd[k[8:]] = v
            msg = model.load_state_dict(nd, strict=False)
            print('*'*50)
            print(msg)
    model.eval()

    train_features  = extract_feature_pipeline(model, args)
    
    print("Features are ready!\nStart the NN eval!")
    closek_info, botk_info = gen_nn_indices(train_features.cuda(), k=384)
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, 'closek_info.npy'), closek_info) 
    np.save(os.path.join(args.output, 'fark_info.npy'), botk_info) 
