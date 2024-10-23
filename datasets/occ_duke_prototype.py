# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import re
import pandas as pd
import os.path as osp

from .base_dataset import BaseDataset


class OCC_DukeMTMCreID(BaseDataset):
    """
    DukeMTMC-reID
    Reference:
    1. Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
    2. Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.
    URL: https://github.com/layumi/DukeMTMC-reID_evaluation

    Dataset statistics:
    # identities: 1404 (train + query)
    # images:16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """

    def __init__(self, cfg, verbose=True, pid_begin=0, **kwargs):
        super(OCC_DukeMTMCreID, self).__init__()
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.DIR)
        # self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.pid_begin = pid_begin

        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.load_data_statistics()
        
        if verbose:
            print("=> Occluded DukeMTMC-reID loaded")
            self.print_dataset_statistics()

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        df = pd.DataFrame(img_paths, columns=['img_path'])
        df['img_path'] = df['img_path'].apply(lambda x: osp.relpath(x, dir_path))
        df['pid'] = df['img_path'].apply(lambda x: int(x.split('_')[0]))
        
        # Extract camera ID from the image path
        df['camid'] = df['img_path'].apply(lambda x: int(x.split('_')[1][1]))
        
        # Update image paths to include the directory path
        df['img_path'] = df['img_path'].apply(lambda x: osp.join(dir_path, x))
        
        # Adjust PID and camera ID
        df['camid'] -= 1
        
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(df['pid'].unique())}
            df['pid'] = df['pid'].map(pid2label)

        df['pid'] += self.pid_begin
        # Add a trackid column with default value 1
        df['trackid'] = 1          
        
        return df
        