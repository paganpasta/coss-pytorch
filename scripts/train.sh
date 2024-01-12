
#Training on moco_v3 teacher
python train_coss.py --output /outputs/students/imagenet/resnet50_mocov2/resnet18/pc_1/coss/ --l_0 1 --l_1 0.5 --subset 0.01  -a resnet18 -k resnet50 -b 256 --distill /outputs/teachers/imagenet/seed/moco_v2.pth  --s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss --data /data/imagenet/Data/CLS-LOC/train
# python -m torch.distributed.launch --nproc_per_node=1  main_lincls.py

#Training on moco_v2 teacher

#python train_coss.py --output /tmp -a resnet18 -k resnet50 -b 256 --distill /outputs/teachers/imagenet/seed/moco_v2.pth --s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss --data /data/imagenet/Data/CLS-LOC/train
#python train_coss.py --output /tmp -a efficientnet_b0 -k resnet50 -b 256 --distill /outputs/teachers/imagenet/seed/moco_v2.pth --s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss --data /data/imagenet/Data/CLS-LOC/train
#python train_coss.py --output /tmp -a vit_tiny -k resnet50 -b 256 --distill /outputs/teachers/imagenet/seed/moco_v2.pth --s-temp 0.1 --t-temp 0.07 --aug moco_v2 --loss coss --data ../data/imagenet/Data/CLS-LOC/train/
