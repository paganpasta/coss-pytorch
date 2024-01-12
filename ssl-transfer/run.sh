#COSS
python linear.py --model efficientnet_b0 --dataset cifar10   --path /rep/SEED-v2/outputs/students/imagenet/resnet50_mocov2/Loss_fuss-Beta_1.0-Epoch_200_Student_efficientnet_b0_distill-Epoch_resnet50.pth.tar.lincls/model_best.pth.tar --only-knn

#SEED
python linear.py --model efficientnet_b0 --dataset cifar10   --path /rep/SEED-v2/outputs/students/imagenet/resnet50_mocov2/SEED_Teacher_moco_T-Epoch_200_Student_efficientnet_b0_distill_resnet50.pth.tar.lincls/model_best.pth.tar --only-knn

#DISCO
python linear.py --model effb0 --dataset cifar10   --path /outputs/disco/ResNet50-EfficientB0-checkpoint_0199.pth.tar.lincls/model_best.pth.tar --only-knn

#BINGO
python linear.py --model efficientnet_b0 --dataset cifar10   --path /outputs/bingo/E0-T50/unsupervised/lr-0.03_batch-256/ckpt.pth.tar.lincls/model_best.pth.tar --only-knn

