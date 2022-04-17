import torch
import pytorch_lightning as pl

import argparse
import pprint

from src.dataset import RestrictedImagenetDataset
from src.model import get_model
from src.trainer import RestrictedImagenetModel
from src.utils import get_transforms


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train_path",
        type=str,
        default="data/train",
    )
    p.add_argument(
        "--valid_path",
        type=str,
        default="data/val",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=192,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    config = p.parse_args()
    return config


def main(config: argparse.Namespace):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get datasets.
    tr_ds = RestrictedImagenetDataset(config.train_path, transforms=get_transforms(is_train=True))
    vl_ds = RestrictedImagenetDataset(config.valid_path, transforms=get_transforms(is_train=False))

    ## Get dataloaders.
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
    )
    vl_loader = torch.utils.data.DataLoader(
        vl_ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
    )

    ## Get a model with randomly initialized weights.
    model = RestrictedImagenetModel()

    ## Set trainer.
    trainer = pl.Trainer(
        gpus=1,
        precision=16,               ## mixed precision; float16
        max_epochs=config.epochs,   ## number of epochs
    )

    ## And just fit.
    trainer.fit(model, tr_loader, vl_loader)
    

if __name__ == "__main__":
    config = define_argparser()
    main(config)
