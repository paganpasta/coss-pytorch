from torchvision.models.alexnet import *
from torchvision.models.vgg import *
from torchvision.models.squeezenet import *
from torchvision.models.inception import *
from torchvision.models.densenet import *
from torchvision.models.googlenet import *
from .architectures.mobilenet.mobilenetv3 import *
from .architectures.efficientnet import *
from .architectures.moco.moco_resnet import *
from .architectures.swav.swav_resnet import *
from .architectures.simclr.simclr_resnet import *
from .architectures.vit.vit import vit_tiny
#from .architectures.eff_disco.efficientnet import efficientnet_b0 as effb0

'''
def clip_resnet50(**kwargs):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='yfcc15m')
    visual = model.visual
    return visual

def clip_vit(**kwargs):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
    visual = model.visual
    return visual

def clip_resnet18(**kwargs):
    from torchvision.models.resnet import resnet18
    model = resnet18(num_classes=kwargs.get('num_classes', 1024))
    return model

'''

