import random
from pathlib import Path
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg
from utils import set_seeds
from datasets import make_dataloader
from models import get_model
from processors.processor_transformer import ProcessorTransformer
from loss import CenterLoss, MultipleLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/vit_base.yml", help="path to config file", type=str
    )

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    # cfg.freeze()

    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks = make_dataloader(cfg)

    loss_fn = MultipleLoss(cfg)
    
    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = number_of_cameras
    cfg.DATASETS.NUMBER_OF_TRACKS = number_of_camera_tracks

    model = get_model(cfg)

    cfg.DEVICE = device

    center_criterion = CenterLoss(cfg)  # center loss

    set_seeds()
    print(cfg)

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # print(model)

    proc = ProcessorTransformer(cfg, 
                                model, 
                                train_loader, 
                                test_loader,
                                torch.optim.Adam(model.parameters(), lr=0.001),
                                center_criterion,
                                loss_fn,
                                epochs=10)
    proc.train()




