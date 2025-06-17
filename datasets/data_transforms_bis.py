import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from timm.data.random_erasing import RandomErasing

class TransformsManager:
    def __init__(self, cfg):
        self.config = cfg
        self._image_train_transforms = None
        self._image_test_transforms = None
        self.train_transforms_cfg = cfg.DATALOADER.TRAIN_TRANSFORMS
        self.test_transforms_cfg = cfg.DATALOADER.TEST_TRANSFORMS
    
    @property
    def image_train_transforms(self):
        
        if self._image_train_transforms is None:
            transforms = []
            # Enumerate keys and values
            for key, value in self.train_transforms_cfg.items():
                # print(f"{i}: {key} = {value}")
                transforms.append(self.create_transforms2(key, value, self.train_transforms_cfg))
            # for transform in self.config.DATALOADER.TRAIN_TRANSFORMS:
            #     transforms.append(self.create_transforms(transform))        
            self._image_train_transforms = T.Compose(transforms)
        return self._image_train_transforms
    
    @property
    def image_test_transforms(self):

        if self._image_test_transforms is None:
            transforms = []
            for key, value in self.test_transforms_cfg.items():
                # print(f"{i}: {key} = {value}")
                transforms.append(self.create_transforms2(key, value, self.test_transforms_cfg, for_test=True))

            # for transform in self.config.DATALOADER.TEST_TRANSFORMS:
            #     transforms.append(self.create_transforms(transform,  test=True))
            self._image_test_transforms = T.Compose(transforms)
        return self._image_test_transforms

    def create_transforms2(self, key, value, transform_cfg, for_test=False):
        
        if key == 'RESIZE' and value == True:
            input_size = self.config.INPUT.SIZE_TEST if for_test else self.config.INPUT.SIZE_TRAIN
            return T.Resize(input_size, interpolation=InterpolationMode.BICUBIC)
        elif key == 'RANDOM_H_FLIP' and value == True:
            return T.RandomHorizontalFlip(p=transform_cfg.RANDOM_H_FLIP_PROB)
        elif key == 'PAD' and value == True:
            return T.Pad(transform_cfg.PADDING)
        elif key == 'RANDOM_CROP' and value == True:
            return T.RandomCrop(self.config.INPUT.SIZE_TRAIN)
        elif key == 'TO_TENSOR' and value == True:
            return T.ToTensor()
        elif key == 'NORMALIZE' and value == True:
            return T.Normalize(mean=self.config.INPUT.PIXEL_MEAN, std=self.config.INPUT.PIXEL_STD)
        elif key == 'RANDOM_ERASING' and value == True:
            return RandomErasing(probability=transform_cfg.RANDOM_ERASING_PROB, mode='pixel', max_count=1, device='cpu')
        # else:
        #     raise ValueError(f"Invalid transform key: {key} with value: {value}")

    # def create_transforms(self, transform, test=False):

    #     transform_name = transform['tranform']

    #     if transform_name == 'resize':
    #         input_size = self.config.INPUT.SIZE_TEST if test else self.config.INPUT.SIZE_TRAIN
    #         return T.Resize(input_size, interpolation=InterpolationMode.BICUBIC)
    #     elif transform_name == 'random_horizontal_flip':
    #         return T.RandomHorizontalFlip(p=transform['prob'])
    #     elif transform_name == 'pad':
    #         return T.Pad(transform['padding'])
    #     elif transform_name == 'random_crop':
    #         return T.RandomCrop(self.config.INPUT.SIZE_TRAIN)
    #     elif transform_name == 'to_tensor':
    #         return T.ToTensor()
    #     elif transform_name == 'normalize':
    #         return T.Normalize(mean=self.config.INPUT.PIXEL_MEAN, std=self.config.INPUT.PIXEL_STD)
    #     elif transform_name == 'random_erasing':
    #         return RandomErasing(probability=transform['prob'], mode='pixel', max_count=1, device='cpu')
    #     else:
    #         raise ValueError("Invalid transform name")
    