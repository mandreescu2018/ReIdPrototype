# make a dataloader for Market1501 dataset
# make a dataloader for DukeMTMC dataset
# make a dataloader for MSMT17 dataset
# make a dataloader for CUHK03 dataset
# make a dataloader for VIPeR dataset

import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import datasets
from .market1501 import Market1501
from .base_dataset import ImageDataset

__factory = {
    'market1501': Market1501, # datasets.Market1501,
    'dukemtmc': None, # datasets.DukeMTMC,
    'msmt17': None, # datasets.MSMT17
    'cuhk03': None, # datasets.CUHK03
    'viper': None, # datasets.VIPeR

}

def make_dataloader(cfg):
    
    dataset = __factory[cfg.DATASETS.NAMES](cfg)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    track_view_num = dataset.num_train_vids

    train_transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    test_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    train_dataset = ImageDataset(dataset.train, transform=train_transform)
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = ImageDataset(dataset.query + dataset.gallery, transform=test_transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    

    return train_dataloader, test_dataloader, num_classes, cam_num, track_view_num



