import torch
import argparse
from config import cfg
from utils import set_seeds, setup_logger
from datasets import make_dataloader
# from datasets.make_dataloader_trans import make_dataloader

def calculate_mean_std(dataloader):
    mean = 0.
    std = 0.
    total_images_count = 0
    # img, pid, camid, camids, target_view, imgpath
    for images, _, _, _, _, _ in dataloader:  # Assuming (images, labels) are returned by the dataloader
        batch_samples = images.size(0)  # batch size (the number of images in the batch)
        images = images.view(batch_samples, images.size(1), -1)  # Reshape into (batch_size, num_channels, height*width)
        mean += images.mean(2).sum(0)  # Sum the mean over batch dimension and spatial dimension
        std += images.std(2).sum(0)  # Sum the std over batch dimension and spatial dimension
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    
    return mean, std

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

    res = calculate_mean_std(test_loader)
    print("Mean:", res[0])
    print("Std:", res[1])

    # train
    # Mean: tensor([-0.2062, -0.2507, -0.2588])
    # Std: tensor([0.4198, 0.4056, 0.4026])

