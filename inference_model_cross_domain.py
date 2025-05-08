import time
import torch
import argparse
from config import cfg
from utils import set_seeds
from datasets import make_dataloader
from models import ModelLoader
from processors.processor_standard import ProcessorStandard


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="configurations/bag_of_tricks/Market/bag_of_tricks.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    set_seeds(cfg.SOLVER.SEED)

    # datasets related
    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)

    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = number_of_cameras
    cfg.DATASETS.NUMBER_OF_TRACKS = number_of_camera_tracks
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num
    
    # Model
    cfg.MODEL.PRETRAIN_CHOICE = 'cross_domain'
    model_loader = ModelLoader(cfg)
    model_loader.load_param()
    
    proc = ProcessorStandard(cfg, 
                                model_loader.model, 
                                train_loader, 
                                test_loader)
    
    start = time.perf_counter()
    
    proc.inference()

    print(f"Time taken: {time.perf_counter() - start}")


