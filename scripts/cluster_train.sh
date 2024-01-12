python -m torch.distributed.launch --master_port 6655 --nproc_per_node=1  train_sampled_coss.py --distributed  \
	--output /outputs/students/imagenet/resnet50_mocov2/resnet18/pc_10/close_exact_batch64/ \
	--l_0 1 --l_1 0.5 --subset 0.1 \
	-a resnet18 -k resnet50 -b 256 --distill /outputs/teachers/imagenet/seed/moco_v2.pth  \
	--s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss \
	--data /data/imagenet/Data/CLS-LOC/train  \
	--closek-info ./weights/clustering/subset_1pc/mocov2_resnet50/clusters_64_subset_0.01_euclidean.pth \
	--sampler cluster




