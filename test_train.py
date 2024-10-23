import torch
import argparse
from config import cfg
from utils import set_seeds, setup_logger
# from datasets import make_dataloader
from models import get_model
from solver import create_scheduler
from processors.processor_transformer import ProcessorTransformer
from loss import MultipleLoss, CenterLoss
from solver.make_optimizer import make_optimizer

from datasets.make_dataloader import make_dataloader

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

    start_epoch = 0

    # logger
    logger = setup_logger("ReIDPrototype", cfg.OUTPUT_DIR, if_train=True)
    logger.info(f"Using {cfg.DEVICE} device")
    logger.info(f"Using {args.config_file} as config file")
    logger.info(f"Saving model in the path :{cfg.OUTPUT_DIR}")
    logger.info(cfg)

    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)
    
    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = number_of_cameras
    cfg.DATASETS.NUMBER_OF_TRACKS = number_of_camera_tracks
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num

    # Model
    model = get_model(cfg)
    # model.to(cfg.DEVICE)
    # print(model)

    # Losses
    loss_fn = MultipleLoss(cfg)
    if cfg.LOSS.CENTER_LOSS:
        center_criterion = CenterLoss(cfg)  # center loss
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    # loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    # loss_fn = make_loss(cfg, num_classes=num_classes)
    # center_criterion2 = CenterLoss(cfg)

    # Optimizers
    optimizer = make_optimizer(cfg, model)
    
    scheduler = create_scheduler(cfg, optimizer)    

    proc = ProcessorTransformer(cfg, 
                                model, 
                                train_loader, 
                                test_loader,
                                optimizer,
                                optimizer_center,
                                center_criterion,
                                loss_fn,
                                scheduler,
                                start_epoch=start_epoch)
    proc.train()




