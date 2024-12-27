import torch
import torch.nn as nn
from datasets import make_dataloader
from models import ModelLoader
from config import cfg
from utils import set_seeds
import numpy as np
import matplotlib.pyplot as plt

def display_weights(model):
    fig,ax = plt.subplots(1,2,figsize=(15,4))

    for p in model.named_parameters():
        if not 'bn1' in p[0] :
            continue
        # get the data and compute their histogram
        thesedata = p[1].data.cpu().numpy().flatten()
        y,x = np.histogram(thesedata,10)

        # for the bias
        if 'bias' in p[0]:
            ax[0].plot((x[1:]+x[:-1])/2,y/np.sum(y),label='%s bias (N=%g)'%(p[0][:-5],len(thesedata)))

        # for the weights
        elif 'weight' in p[0]:
            ax[1].plot((x[1:]+x[:-1])/2,y/np.sum(y),label='%s weight (N=%g)'%(p[0][:-7],len(thesedata)))



    ax[0].set_title('Biases per layer')
    ax[0].legend()
    ax[1].set_title('Weights per layer')
    ax[1].legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.show()

def view_weights(model):
    for name, param in model.named_parameters():
        print(name, param.size())

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/bag_of_tricks/Market/bag_of_tricks.yml", help="path to config file", type=str
    )

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # datasets related
    train_loader, test_loader, num_classes, number_of_cameras, number_of_camera_tracks, query_num = make_dataloader(cfg)

    cfg.DATASETS.NUMBER_OF_CLASSES = num_classes
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = query_num

    # Model
    cfg.MODEL.PRETRAIN_CHOICE = 'test'
    model_loader = ModelLoader(cfg)
    model_loader.load_param()

    # view_weights(model_loader.model)
    display_weights(model_loader.model)

    

    

    

    