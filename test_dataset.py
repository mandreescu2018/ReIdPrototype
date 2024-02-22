import argparse
from datasets import make_dataloader
from config import cfg

def main(cfg):
    print("dataset:", cfg.DATASETS.NAMES)
    train_loader, test_loader = make_dataloader(cfg)
    print("train loader dim:", len(train_loader.dataset))
    print("test loader dim:", len(test_loader.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test dataset")
    parser.add_argument("--config", type=str, default='configurations/main.yml', help="Path to the configuration file")
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config)

    main(cfg)


