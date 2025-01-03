import os
import glob
import re
from torch.utils.data import Dataset
from .base_dataset import BaseDataset


import os
import glob
import re
import pandas as pd

class Market1501(BaseDataset):
    def __init__(self, cfg, verbose=True, pid_begin=0):
        self.dataset_dir = os.path.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.DIR)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self.pid_begin = pid_begin
        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.load_data_statistics()

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics()

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid, camid = map(int, match.groups())
                if pid == -1: continue  # junk images are ignored
                data.append((img_path, pid, camid))

        df = pd.DataFrame(data, columns=['img_path', 'pid', 'camid'])
        df = df[df['pid'] != -1]  # Remove junk images
        df['camid'] -= 1  # index starts from 0

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(df['pid'].unique())}
            df['pid'] = df['pid'].map(pid2label)

        df['pid'] += self.pid_begin
        df['trackid'] = 0  # Add a trackid column with default value 0
        return df