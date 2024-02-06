#python3 train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 3 --resume MODEL.WEIGHTS ./weights/bingo_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/pascal_voc/moco_v2/resnet18/bingo
#python3 train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 3 --resume MODEL.WEIGHTS ./weights/coss_run2_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/pascal_voc/moco_v2/resnet18/coss
#python3 train_net.py --config-file coco_voc/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/cfcoss_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/pascal_voc/moco_v2/resnet18/cf_coss
#python3 train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 3 --resume MODEL.WEIGHTS ./weights/disco_r19.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/pascal_voc/moco_v2/resnet18/disco
#python3 train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 3 --resume MODEL.WEIGHTS ./weights/smd_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/pascal_voc/moco_v2/resnet18/smd
#python3 train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 3 --resume MODEL.WEIGHTS ./weights/seed_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/pascal_voc/moco_v2/resnet18/seed

#MOCO-v3 PASCAL VOC
python3 train_net.py --config-file coco_voc/pascal_voc_R_50_C4_12k_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS /workspace/bingo/detection/weights/mocov3_r18_cfcoss.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/pascal_voc07/moco_v3/resnet18/cf_coss


#MOCO-v2 COCO

#python3 train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/bingo_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/coco/moco_v2/resnet18/bingo
#python3 train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/coss_run2_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/coco/moco_v2/resnet18/coss
#python3 train_net.py --config-file coco_voc/coco_R_50_C4_2x_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/cfcoss_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/coco/moco_v2/resnet18/cf_coss
#python3 train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/disco_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/coco/resnet18/disco
#python3 train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/smd_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/coco/moco_v2/resnet18/smd
#python3 train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/seed_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/coco/moco_v2/resnet18/seed

#MOCO-v3 COCO
#python3 train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 4 --resume MODEL.WEIGHTS ./weights/mocov3_coss_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR ./output/bingo_config/coco/moco_v3/resnet18/coss

#MOCO-v2 CITYSCAPES
#python3 train_net.py --config-file cityscapes/mask_rcnn_R_50_FPN.yaml --dist-url 'tcp://127.0.0.1:52111' --num-gpus 4 --resume MODEL.WEIGHTS /workspace/bingo/detection/weights/bingo_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR /outputs/cityscapes/moco_v2/resnet18/bingo
#python3 train_net.py --config-file cityscapes/mask_rcnn_R_50_FPN.yaml --dist-url 'tcp://127.0.0.1:52111' --num-gpus 4 --resume MODEL.WEIGHTS /workspace/bingo/detection/weights/coss_run2_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR /outputs/cityscapes/moco_v2/resnet18/coss
#python3 train_net.py --config-file cityscapes/mask_rcnn_R_50_FPN.yaml --dist-url 'tcp://127.0.0.1:52111' --num-gpus 4 --resume MODEL.WEIGHTS /workspace/bingo/detection/weights/disco_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR /outputs/cityscapes/moco_v2/resnet18/disco
#python3 train_net.py --config-file cityscapes/mask_rcnn_R_50_FPN.yaml --dist-url 'tcp://127.0.0.1:52111' --num-gpus 4 --resume MODEL.WEIGHTS /workspace/bingo/detection/weights/smd_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR /outputs/cityscapes/moco_v2/resnet18/smd
#python3 train_net.py --config-file cityscapes/mask_rcnn_R_50_FPN.yaml --dist-url 'tcp://127.0.0.1:52111' --num-gpus 4 --resume MODEL.WEIGHTS /workspace/bingo/detection/weights/seed_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR /outputs/cityscapes/moco_v2/resnet18/seed

#MOCO-v3 CITYSCAPES
#python3 train_net.py --config-file cityscapes/mask_rcnn_R_50_FPN.yaml --dist-url 'tcp://127.0.0.1:52111' --num-gpus 4 --resume MODEL.WEIGHTS /workspace/bingo/detection/weights/mocov3_coss_r18.pkl  MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 OUTPUT_DIR /outputs/cityscapes/moco_v3/resnet18/coss
