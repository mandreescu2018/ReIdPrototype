import argparse
from datasets import make_dataloader
from datasets.make_dataloader_prototype import make_dataloader_prototype
from config import cfg
import matplotlib.pyplot as plt

def get_classic_dataloader(cfg):
    print("dataset:", cfg.DATASETS.NAMES)
    train_loader, test_loader, number_of_classes, number_of_cameras, number_of_camera_tracks, number_of_query_imgs = make_dataloader(cfg)
    print("train loader dim:", len(train_loader.dataset))
    print("test loader dim:", len(test_loader.dataset))
    print("classes:", number_of_classes)
    print("number of cameras:", number_of_cameras)
    print("number of camera tracks:", number_of_camera_tracks)
    print("number of query images:", number_of_query_imgs)

def get_prototype_dataloader(cfg):
    print("dataset:", cfg.DATASETS.NAMES)
    train_loader, test_loader, number_of_classes, number_of_cameras, number_of_camera_tracks, number_of_query_imgs = make_dataloader_prototype(cfg)
    print("train loader dim:", len(train_loader.dataset))
    print("test loader dim:", len(test_loader.dataset))
    print("classes:", number_of_classes)
    print("number of cameras:", number_of_cameras)
    print("number of camera tracks:", number_of_camera_tracks)
    print("number of query images:", number_of_query_imgs)
    
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(test_loader):
    # for n_iter, (img, vid, target_cam, target_view) in enumerate(val_loader):
        if n_iter < 5:
            print(f'Image: {img.shape}, camids: {camids.shape}, target_view: {target_view.shape}')
            plt.imshow(img[0].permute(1,2,0))
            plt.show()

        else:
            break

def main(cfg):
    # get_classic_dataloader(cfg)
    get_prototype_dataloader(cfg)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test dataset")
    parser.add_argument("--config_file", type=str, default='configurations/vit_base.yml', help="Path to the configuration file")
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    main(cfg)


