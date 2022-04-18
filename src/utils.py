import albumentations
from albumentations.pytorch import ToTensorV2


def get_transforms(is_train: bool = True):
    if is_train:
        transforms = albumentations.Compose([
            albumentations.Resize(256, 256),
            albumentations.RandomCrop(224, 224),
            albumentations.Normalize(), ## standarization
            ## And some augmentations
            ToTensorV2(),
        ])

    else:
        transforms = albumentations.Compose([
            albumentations.Resize(256, 256),
            albumentations.RandomCrop(224, 224),
            albumentations.Normalize(), ## standarization
            ## And no augmentations
            ToTensorV2(),
        ])
    
    return transforms


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad != None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data ** norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data ** norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
