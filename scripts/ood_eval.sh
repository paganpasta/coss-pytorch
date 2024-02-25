echo "BINGO"

python -m torch.distributed.launch lincls.py --arch resnet18  --evaluate  --resume model_best.pth.tar  --data /data/imagenetv2-matched-frequency-format-val/val/  --subset 1.0
python -m torch.distributed.launch lincls.py --arch resnet18  --evaluate  --resume model_best.pth.tar  --data /data/imagenetv2-threshold0.7-format-val/  --subset 1.0
python -m torch.distributed.launch lincls.py --arch resnet18  --evaluate  --resume model_best.pth.tar  --data /data/imagenetv2-top-images-format-val/ --subset 1.0
python -m torch.distributed.launch lincls.py --arch resnet18  --evaluate  --resume model_best.pth.tar  --data /data/imagenet-sketch/val --subset 1.0
python -m torch.distributed.launch lincls.py --arch resnet18  --evaluate  --resume model_best.pth.tar  --data /data/imagenet-c/ --subset 1.0


