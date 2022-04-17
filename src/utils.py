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
