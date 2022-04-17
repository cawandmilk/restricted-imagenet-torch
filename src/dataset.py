from cv2 import IMREAD_COLOR
import torch
import pytorch_lightning as pl

import cv2
import tqdm

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Dict


RESTRICTED_IMAGNET_RANGES = [
    (151, 268),     ## dog (117)
    (281, 285),     ## cat (5) 
    ( 30,  32),     ## frog (3)
    ( 33,  37),     ## turtle (5)
    ( 80, 100),     ## bird (21)
    (365, 382),     ## monkey (14)
    (389, 397),     ## fish (9)
    (118, 121),     ## crab (4)
    (300, 319),     ## insect (20)
]


class RestrictedImagenetDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        fpath: str,
        transforms = None,
        **kwargs,
    ):
        super(RestrictedImagenetDataset, self).__init__(**kwargs)

        self.fpath = Path(fpath)
        self.transforms = transforms
        
        ## Load preprocessed images and labels.
        self.df = self._load_data()


    def _get_restricted_imagenet_pairs(self, fpath: str, ranges: List[tuple] = RESTRICTED_IMAGNET_RANGES) -> tuple:
        range_sets = [set(range(s, e+1)) for s,e in ranges]

        ## inp: input image path (e.g., ./data/val/n01440764/ILSVRC2012_val_00000293.JPEG)
        ## tar: label of the image (e.g., 0)
        inp, tar = [], []

        for idx, dir_name in enumerate(sorted(Path(fpath).iterdir())):
            for new_idx, range_set in enumerate(range_sets):
                if idx in range_set:
                    ## Get all image path.
                    inp_ = [str(i) for i in sorted(list(Path(dir_name).glob("*.JPEG")))]
                    tar_ = [new_idx] * len(inp_)

                    ## And accumulate it.
                    inp.extend(inp_)
                    tar.extend(tar_)

                    ## We don't look the other range sets.
                    break

        return np.array(inp), np.array(tar)


    def _load_data(self) -> pd.DataFrame:
        ## Get input and target pairs.
        inp, tar = self._get_restricted_imagenet_pairs(self.fpath)
            
        ## Print the sizes.
        sz = sum([Path(i).stat().st_size for i in inp]) / (2**3 * 2**30) ## GB
        print(f"[{self.fpath}] # of images (*.JPEG): {len(inp)}, size: {sz:.2f}GB")

        ## Make dataframe.
        df = pd.DataFrame({"inp": inp, "tar": tar})
        print(f"[{self.fpath}] all data loaded")

        return df

    
    def __len__(self) -> int:
        return self.df.shape[0]


    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        ## Get instance from dataframe.
        instance = self.df.iloc[idx]
        return_values = {"inp": instance["inp"], "tar": instance["tar"]}

        ## Load images.
        return_values["inp"] = cv2.imread(return_values["inp"], cv2.IMREAD_COLOR)

        ## Apply transforms.
        if self.transforms != None:
            return_values["inp"] = self.transforms(image=return_values["inp"])["image"]

        ## And export.
        return return_values


class RestrictedImagenetDataloader(pl.LightningDataModule):
    pass