import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing

class Transforms:
    def __init__(self, cfg):
        
        self.train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.HF_PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            # T.RandomErasing(p=cfg.INPUT.RE_PROB),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
        
        self.test_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
        
        self.video_train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.Pad(10),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            # T.RandomErasing(p=0.5, scale=erase_scale, ratio=erase_ratio)

            ])
        self.video_test_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])        
        

    def get_train_transforms(self):
        return self.train_transforms

    def get_test_transforms(self):
        return self.test_transforms