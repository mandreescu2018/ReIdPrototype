import torch
import torch.nn as nn
from loss.loss_factory import LossComposer
from config import cfg

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/MobileNet/Market/mobilenet.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Define loss functions
    # loss_composer = DynamicLossComposer(cfg)
    # loss_composer.instantiate_losses()
    # print(loss_composer.loss_functions)

    