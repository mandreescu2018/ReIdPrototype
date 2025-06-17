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
    