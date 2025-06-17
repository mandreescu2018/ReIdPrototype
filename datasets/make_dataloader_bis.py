import os
from functools import partial

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import datasets
from .market1501 import Market1501
from .msmt17_prototype import MSMT17_Prototype
from .occ_duke_prototype import OCC_DukeMTMCreID
from .dukemtmcreid import DukeMTMCreID
from .image_dataset import ImageDataset
from .data_transforms import TransformsManager
from .sampler import RandomIdentitySampler
import pandas as pd



class CustomCollate:
    
    def __init__(self, cfg):
       
        self.stack_imgs = partial(torch.stack, dim=0)
        self.config = cfg

    def apply_transform(self, object, image=False):
        if image:
            return self.stack_imgs(object)
        else:
            return torch.tensor(object, dtype=torch.int64)

    def train_collate_fn(self, batch):
        
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        
        imgs = self.apply_transform(imgs, image=True)
        pids = self.apply_transform(pids)
        camids = self.apply_transform(camids)
        viewids = self.apply_transform(viewids)

        return imgs, pids, camids, viewids, img_paths

    def val_collate_fn(self, batch):
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        
        imgs = self.apply_transform(imgs, image=True)
        camids_tensor = self.apply_transform(camids)
        viewids = self.apply_transform(viewids)

        return imgs, pids, camids, camids_tensor, viewids, img_paths

class ReIDDataLoader:
    __factory = {
        'market1501': Market1501,
        'dukemtmc': DukeMTMCreID, 
        'msmt17': MSMT17_Prototype, 
        'cuhk03': None, # datasets.CUHK03
        'occ_duke': OCC_DukeMTMCreID, 
    }
    def __init__(self, cfg):
        self.cfg = cfg
        self._train_dataset = None
        self._validation_dataset = None
        self.num_workers = cfg.DATALOADER.NUM_WORKERS
        
        self.transforms_manager = TransformsManager(cfg)
        self.custom_collate = CustomCollate(cfg)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = ReIDDataLoader.__factory[self.cfg.DATASETS.NAMES](self.cfg)
        return self._train_dataset

    @property
    def validation_set(self):
        if self._validation_dataset is None:
            if self.cfg.DATASETS.TEST is not None:
                val_set = ReIDDataLoader.__factory[self.cfg.DATASETS.TEST](self.cfg)
            else:
                val_set = self.train_dataset
            self._validation_dataset = ImageDataset(pd.concat([val_set.query, val_set.gallery], ignore_index=True), self.val_transforms)
        return self._validation_dataset
        
    @property
    def train_transforms(self):
        return self.transforms_manager.image_train_transforms
    
    @property
    def val_transforms(self):
        return self.transforms_manager.image_test_transforms

    @property
    def train_dataloader(self):
        train_dataset = ImageDataset(self.train_dataset.train, transform=self.train_transforms)
        
        return DataLoader(
            train_dataset,
            batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
            num_workers=self.num_workers,
            sampler=RandomIdentitySampler(self.train_dataset.train.itertuples(index=False, name=None), self.cfg.SOLVER.IMS_PER_BATCH, self.cfg.DATALOADER.NUM_INSTANCE),
            collate_fn=self.custom_collate.train_collate_fn,
        )

    @property
    def val_loader(self):
        return DataLoader(
            self.validation_set, 
            batch_size=self.cfg.TEST.IMS_PER_BATCH, 
            shuffle=False,   
            num_workers=self.num_workers,
            collate_fn=self.custom_collate.val_collate_fn
        )

    @property
    def num_classes(self):
        return self.train_dataset.num_train_pids

    @property
    def cameras_number(self):
        return self.train_dataset.num_train_cams

    @property
    def track_view_num(self):
        return self.train_dataset.num_train_vids

    @property
    def query_num(self):
        return len(self.train_dataset.query)
