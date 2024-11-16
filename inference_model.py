import torch
import argparse
from config import cfg
from utils import set_seeds
from datasets import make_dataloader
# from datasets.make_dataloader_trans import make_dataloader
from models import ModelLoader
from processors.processor_transformer import ProcessorTransformer


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="configurations/vit_base.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    set_seeds(cfg.SOLVER.SEED)

    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # datasets related
    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)

    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num
    
    # Model
    model_loader = ModelLoader(cfg)
    model, _, _, _ = model_loader.load_param()
    # model = get_model(cfg)
    
    proc = ProcessorTransformer(cfg, 
                                model, 
                                train_loader, 
                                test_loader)
    proc.inference()


