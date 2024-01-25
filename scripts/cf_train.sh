#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  train_sampled_coss.py --distributed  \
#	--output /outputs/students/imagenet/resnet50_mocov2/resnet34/pc_100/cf_coss/close_exactK_batch64_nearest15/ \
#	--l_0 1 --l_1 1  \
#	-a resnet34 -k resnet50 -b 64 --distill /outputs/teachers/imagenet/seed/moco_v2.pth  \
#	--s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss \
#	--data /data/imagenet/Data/CLS-LOC/train  \
#	--closek-info ./weights/resnet50-mocov2/imagenet_subset_100pc/closek_info.npy \
#	-p 1000 --sampler cf  --nearest-k 15 --total-k 31 --epochs 25


#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4  train_sampled_coss.py --distributed  \
#	--output /outputs/students/imagenet/resnet50_mocov2/eff-b0/pc_100/cf_coss/close_exactK_batch64_nearest15/ \
#	--l_0 1 --l_1 1  \
#	-a efficientnet_b0 -k resnet50 -b 64 --distill /outputs/teachers/imagenet/seed/moco_v2.pth  \
#	--s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss \
#	--data /data/imagenet/Data/CLS-LOC/train  \
#	--closek-info ./weights/resnet50-mocov2/imagenet_subset_100pc/closek_info.npy \
#	-p 1000 --sampler cf  --nearest-k 15 --total-k 31 --epochs 25

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4  train_sampled_coss.py --distributed  \
	--output /outputs/students/imagenet/resnet50_mocov3/resnet18/pc_100/cf_coss/close_exactK_batch64_nearest15/ \
	--l_0 1 --l_1 1  \
	-a resnet18 -k resnet50 -b 64 --distill /outputs/teachers/imagenet/moco-v3/pixel_mocov3.pth  \
	--s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss \
	--data /data/imagenet/Data/CLS-LOC/train  \
	--closek-info ./weights/resnet50-mocov3/imagenet_subset_100pc/closek_info.npy \
	-p 1000 --sampler cf  --nearest-k 15 --total-k 31 --epochs 25

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  train_sampled_coss.py --distributed  \
	--output /outputs/students/imagenet/resnet50_mocov2/resnet18/pc_100/cf_coss/close_exactK_batch64_nearest15_epochs13/ \
	--l_0 1 --l_1 1  \
	-a resnet18 -k resnet50 -b 64 --distill /outputs/teachers/imagenet/seed/moco_v2.pth  \
	--s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss \
	--data /data/imagenet/Data/CLS-LOC/train  \
	--closek-info ./weights/resnet50-mocov2/imagenet_subset_100pc/closek_info.npy \
	-p 1000 --sampler cf  --nearest-k 15 --total-k 31 --epochs 13

