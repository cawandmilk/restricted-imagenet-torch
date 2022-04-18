import torch
import torchvision

def get_model(pretrained: bool = False, num_classes: int = 9):
    ## Ref:
    ##  - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    ##  - https://pytorch.org/hub/pytorch_vision_resnet/
    ##  - https://pytorch.org/vision/stable/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    model = torchvision.models.resnet50(pretrained=pretrained, num_classes=num_classes)
    
    return model
