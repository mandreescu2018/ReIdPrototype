import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from timm.data.random_erasing import RandomErasing

class TransformsManager:
    def __init__(self, cfg):
        self.config = cfg
        self._image_train_transforms = None
        self._image_test_transforms = None
    
    @property
    def image_train_transforms(self):
        
        if self._image_train_transforms is None:
            transforms = []
            for transform in self.config.DATALOADER.TRAIN_TRANSFORMS:
                transforms.append(self.create_transforms(transform))        
            self._image_train_transforms = T.Compose(transforms)
        return self._image_train_transforms
    
    @property
    def image_test_transforms(self):

        if self._image_test_transforms is None:
            transforms = []
            for transform in self.config.DATALOADER.TEST_TRANSFORMS:
                transforms.append(self.create_transforms(transform,  test=True))
            self._image_test_transforms = T.Compose(transforms)
        return self._image_test_transforms

    def create_transforms(self, transform, test=False):

        transform_name = transform['tranform']

        if transform_name == 'resize':
            input_size = self.config.INPUT.SIZE_TEST if test else self.config.INPUT.SIZE_TRAIN
            return T.Resize(input_size, interpolation=InterpolationMode.BICUBIC)
        elif transform_name == 'random_horizontal_flip':
            return T.RandomHorizontalFlip(p=transform['prob'])
        elif transform_name == 'pad':
            return T.Pad(transform['padding'])
        elif transform_name == 'random_crop':
            return T.RandomCrop(self.config.INPUT.SIZE_TRAIN)
        elif transform_name == 'to_tensor':
            return T.ToTensor()
        elif transform_name == 'normalize':
            return T.Normalize(mean=self.config.INPUT.PIXEL_MEAN, std=self.config.INPUT.PIXEL_STD)
        elif transform_name == 'random_erasing':
            return RandomErasing(probability=transform['prob'], mode='pixel', max_count=1, device='cpu')
        else:
            raise ValueError("Invalid transform name")
    

# class Transforms:
#     def __init__(self, cfg):
        
#         self.train_transforms = T.Compose([
#             T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=InterpolationMode.BICUBIC),
#             T.RandomHorizontalFlip(p=cfg.INPUT.HF_PROB),
#             T.Pad(cfg.INPUT.PADDING),
#             T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
#             T.ToTensor(),
#             T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
#             # T.RandomErasing(p=cfg.INPUT.RE_PROB),
#             RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
#         ])
        
#         self.test_transforms = T.Compose([
#             T.Resize(cfg.INPUT.SIZE_TEST),
#             T.ToTensor(),
#             T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
#         ])
        
#         self.video_train_transforms = T.Compose([
#             T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
#             T.Pad(10),
#             T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
#             # T.RandomErasing(p=0.5, scale=erase_scale, ratio=erase_ratio)

#             ])
#         self.video_test_transforms = T.Compose([
#             T.Resize(cfg.INPUT.SIZE_TEST),
#             T.ToTensor(),
#             T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
#         ])        
        

#     def get_train_transforms(self):
#         return self.train_transforms

#     def get_test_transforms(self):
#         return self.test_transforms