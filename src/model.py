import torch

def get_model(pretrained: bool = False):
    ## Ref: https://pytorch.org/hub/pytorch_vision_resnet/
    model = torch.hub.load("pytorch/vision:v0.12.0", "resnet50", pretrained=pretrained)
    return model
