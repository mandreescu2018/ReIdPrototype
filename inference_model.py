import torch
import argparse
from config import cfg
from utils import set_seeds
from datasets import make_dataloader
# from datasets.make_dataloader_trans import make_dataloader
from models import ModelLoader
from processors.processor_prototype import ProcessorPrototype


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
    train_loader, test_loader, num_classes, _, _, query_num = make_dataloader(cfg)

    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num
    
    # Model
    cfg.MODEL.PRETRAIN_CHOICE = 'test'
    model_loader = ModelLoader(cfg)
    model_loader.load_param()
    
    proc = ProcessorPrototype(cfg, 
                                model_loader.model, 
                                train_loader, 
                                test_loader)
    proc.inference()


