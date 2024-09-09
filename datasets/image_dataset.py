import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


class ImageDataset_prototype(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            feature_columns = ['img_path', 'pid', 'camid', 'trackid']
            # img_path, pid, camid, trackid = self.dataframe.loc[index, feature_columns].values.astype('float32')
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

class PandasDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column=None, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): The pandas DataFrame containing the data.
            feature_columns (list of str): List of column names that represent the features.
            target_column (str, optional): The column name of the target (label). If None, it assumes no labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features and target from the dataframe
        features = self.dataframe.loc[idx, self.feature_columns].values.astype('float32')
        if self.target_column:
            target = self.dataframe.loc[idx, self.target_column].astype('float32')
        else:
            target = None

        # Apply optional transformation
        if self.transform:
            features = self.transform(features)

        if target is not None:
            return torch.tensor(features), torch.tensor(target)
        else:
            return torch.tensor(features)

# Example Usage:
# Sample DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [6, 7, 8, 9, 10],
    'target': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Create the dataset
dataset = PandasDataset(
    dataframe=df,
    feature_columns=['feature1', 'feature2'],
    target_column='target'
)

# Access a sample
# print(dataset[0])  # Output: (tensor([1., 6.]), tensor(0.))
