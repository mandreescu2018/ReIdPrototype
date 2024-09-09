import torch
import argparse
from config import cfg
from utils import set_seeds, setup_logger
from datasets import make_dataloader
from models import get_model
from solver import create_scheduler
from processors.processor_transformer import ProcessorTransformer
from loss import CenterLoss, MultipleLoss
from solver.make_optimizer import make_optimizer

from datasets.make_dataloader_prototype import make_dataloader_prototype

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

    # logger
    logger = setup_logger("ReIDPrototype", cfg.OUTPUT_DIR, if_train=True)
    logger.info(f"Using {cfg.DEVICE} device")
    logger.info(f"Using {args.config_file} as config file")
    logger.info(f"Saving model in the path :{cfg.OUTPUT_DIR}")
    logger.info(cfg)

    # datasets related
    # train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)
    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader_prototype(cfg)
    
    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = number_of_cameras
    cfg.DATASETS.NUMBER_OF_TRACKS = number_of_camera_tracks
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num

    # Model
    model = get_model(cfg)
    

    # Losses
    loss_fn = MultipleLoss(cfg)
    center_criterion = CenterLoss(cfg)  # center loss

    # Optimizers
    optimizer = make_optimizer(cfg, model)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    scheduler = create_scheduler(cfg, optimizer)

    if cfg.MODEL.PRETRAIN_CHOICE == 'resume':
        model_path = cfg.MODEL.PRETRAIN_PATH        
        print('Loading pretrained model for resume from {}'.format(model_path))
        model, optimizer, current_epoch, scheduler, _ = model.load_param_resume(model_path, optimizer, scheduler)

    proc = ProcessorTransformer(cfg, 
                                model, 
                                train_loader, 
                                test_loader,
                                optimizer,
                                optimizer_center,
                                center_criterion,
                                loss_fn,
                                scheduler)
    proc.train()




