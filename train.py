import torch
import argparse
from config import cfg
from utils import set_seeds
from models import ModelLoader
from processors import get_processor
from loss.loss_factory_prototype import LossComposer
from solver.make_optimizer import OptimizerFactory
from solver import LearningRateScheduler
from functional_logging.stream_logger import StreamLogger
from loss.patch_memory import PatchMemory
from loss import Pedal, PatchMemory

from datasets import ReIDDataLoader, make_dataloader


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/Trans_ReID/Market/vit_base.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    # cfg.freeze()
    set_seeds(cfg.SOLVER.SEED)

    # logger
    stream_logger = StreamLogger(cfg=cfg)
    # logger.info(f"Using {DeviceManager.get_device()} device")
    stream_logger.info(f"Using {args.config_file} as config file")
    stream_logger.info(f"Saving model in the path :{cfg.OUTPUT_DIR}")
    stream_logger.info(cfg)

    # Data Loaders

    data_loaders  = ReIDDataLoader(cfg)    
    train_loader = data_loaders.train_dataloader
    test_loader = data_loaders.val_loader
    
    cfg.DATASETS.NUMBER_OF_CLASSES = data_loaders.num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = data_loaders.cameras_number
    cfg.DATASETS.NUMBER_OF_TRACKS = data_loaders.track_view_num
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = data_loaders.query_num

    # train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)

    # cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    # cfg.DATASETS.NUMBER_OF_CAMERAS = number_of_cameras
    # cfg.DATASETS.NUMBER_OF_TRACKS = number_of_camera_tracks
    # cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num

    # Model
    model_loader = ModelLoader(cfg)
    
    # Losses
    loss_composer = LossComposer(cfg)
    if loss_composer.center_criterion is not None:
        center_criterion = loss_composer.center_criterion
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    else:
        center_criterion = None
        optimizer_center = None

    model_loader.center_criterion = center_criterion
    model_loader.optimizer_center = optimizer_center

    kwargs = {}
    if cfg.MODEL.PC_LOSS:        
        pc_scale = 0.02
        cluster_k = 10
        kwargs["patch_centers"] = PatchMemory(momentum=0.1, num=1)
        kwargs["pc_criterion"] = Pedal(scale=pc_scale, k=cluster_k).cuda()

    # Optimizers
    optimizer_fact = OptimizerFactory(cfg, model_loader.model)
    optimizer = optimizer_fact.make_optimizer()
    scheduler = LearningRateScheduler(optimizer, cfg)
    
    model_loader.optimizer = optimizer
    model_loader.scheduler = scheduler

    model_loader.load_param()

    proc = get_processor(cfg)

    proc = proc(cfg, 
                model_loader.model, 
                train_loader, 
                test_loader,
                model_loader.optimizer,
                model_loader.optimizer_center,
                model_loader.center_criterion,
                loss_composer,
                model_loader.scheduler,
                start_epoch=model_loader.start_epoch,
                **kwargs)
    proc.train()




