# make a dataloader for Market1501 dataset
# make a dataloader for DukeMTMC dataset
# make a dataloader for MSMT17 dataset
# make a dataloader for CUHK03 dataset
# make a dataloader for VIPeR dataset

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
from .data_transforms import Transforms
from .sampler import RandomIdentitySampler
import pandas as pd

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID, 
    'msmt17': MSMT17_Prototype, 
    'cuhk03': None, # datasets.CUHK03
    'occ_duke': OCC_DukeMTMCreID, 
}

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
        
        imgs, pids, camids, viewids, _ = zip(*batch)
        
        # imgs = torch.stack(imgs, dim=0)
        imgs = self.apply_transform(imgs, image=True)
        pids = self.apply_transform(pids)
        camids = self.apply_transform(camids)
        viewids = self.apply_transform(viewids)

        # pids = torch.tensor(pids, dtype=torch.int64)
        # camids = torch.tensor(camids, dtype=torch.int64)
        # viewids = torch.tensor(viewids, dtype=torch.int64)

        return imgs, pids, camids, viewids,

    def val_collate_fn(self, batch):
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        
        imgs = self.apply_transform(imgs, image=True)
        camids_tensor = self.apply_transform(camids)
        viewids = self.apply_transform(viewids)

        return imgs, pids, camids, camids_tensor, viewids, img_paths
            
    
    # def val_collate_fn(self, batch):
    #     imgs, pids, camids, viewids, img_paths = zip(*batch)
    #     viewids = torch.tensor(viewids, dtype=torch.int64)
    #     camids_batch = torch.tensor(camids, dtype=torch.int64)
    #     return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths
    
    

# def train_collate_fn(batch):
#     """
#     # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
#     """
#     imgs, pids, camids, viewids , _ = zip(*batch)
#     pids = torch.tensor(pids, dtype=torch.int64)
#     viewids = torch.tensor(viewids, dtype=torch.int64)
#     camids = torch.tensor(camids, dtype=torch.int64)
#     return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    
    dataset = __factory[cfg.DATASETS.NAMES](cfg)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    track_view_num = dataset.num_train_vids

    transf = Transforms(cfg)
    train_transforms = transf.get_train_transforms()
    val_transforms = transf.get_test_transforms()

    custom_collate = CustomCollate(cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    
    train_dataset = ImageDataset(dataset.train, transform=train_transforms)
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        sampler=RandomIdentitySampler(dataset.train.itertuples(index=False, name=None), cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        collate_fn=custom_collate.train_collate_fn,
    )

    val_set = ImageDataset(pd.concat([dataset.query, dataset.gallery], ignore_index=True), val_transforms)

    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.TEST.IMS_PER_BATCH, 
        shuffle=False,   
        num_workers=num_workers,
        collate_fn=custom_collate.val_collate_fn
    )
    
    return train_dataloader, val_loader, num_classes, cam_num, track_view_num, len(dataset.query)



