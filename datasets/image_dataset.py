import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


class ImageDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            feature_columns = ['img_path', 'pid', 'camid', 'trackid']
            img_path, pid, camid, trackid = self.dataframe.loc[index, feature_columns].values
            img = self.read_image(img_path)
            img = self.transform(img)
            return img, pid, camid, trackid, img_path.split('/')[-1]
        
        @staticmethod
        def read_image(img_path):
            """Keep reading image until succeed.
            This can avoid IOError incurred by heavy IO process."""
            if not os.path.exists(img_path):
                raise IOError(f"{img_path} does not exist")
            while True:
                try:
                    img = Image.open(img_path).convert('RGB')
                    break
                except IOError:
                    print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            return img

