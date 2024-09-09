import torch
import argparse
from config import cfg
from utils import set_seeds, setup_logger
from datasets import make_dataloader
# from datasets.make_dataloader_trans import make_dataloader
from models import get_model
from solver import create_scheduler
from processors.processor_transformer import ProcessorTransformer
from loss import CenterLoss, MultipleLoss
from solver.make_optimizer import make_optimizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/vit_base.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    # cfg.freeze()
    set_seeds(cfg.SOLVER.SEED)

    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # datasets related
    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)
    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num
    model = get_model(cfg)
    
    # checkpoint = torch.load(cfg.TEST.WEIGHT)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_param(cfg.TEST.WEIGHT)

    proc = ProcessorTransformer(cfg, 
                                model, 
                                train_loader, 
                                test_loader)
    proc.inference()


