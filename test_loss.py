import torch
import torch.nn as nn
from loss.loss_factory_prototype import LossComposer
from loss.center_loss import CenterLoss
from config import cfg
from utils import set_seeds

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/bag_of_tricks/Market/bag_of_tricks.yml", help="path to config file", type=str
    )

    # "configurations\bag_of_tricks\Market\bag_of_tricks.yml"
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Define loss functions
    cfg.DATASETS.NUMBER_OF_CLASSES = 751
    loss_composer = LossComposer(cfg)
    # loss_composer.instantiate_losses()
    print(loss_composer.loss_functions)

    set_seeds()

    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    center_loss = CenterLoss(cfg)
    features = torch.rand(16, 2048).to("cuda")
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    targets = targets.to("cuda")

    loss = center_loss(features, targets)
    print(loss)



    

    

    