
import glob
import re

import os.path as osp
import pandas as pd

from.base_dataset_prototype import BaseDataset_prototype


class MSMT17_Prototype(BaseDataset_prototype):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """

    def __init__(self, cfg, verbose=True, pid_begin=0):
        super(MSMT17_Prototype, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.DIR)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        self.train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        self.train = pd.concat([self.train, val], ignore_index=True)
        # pd.concat([dataset.query, dataset.gallery], ignore_index=True)

        self.query = self._process_dir(self.test_dir, self.list_query_path)
        self.gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        
        self.load_data_statistics()

        if verbose:
            print("=> MSMT17 prototype loaded")
            self.print_dataset_statistics()
       
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
    
    def _process_dir(self, dir_path, list_path):
        # Read the list file into a DataFrame
        df = pd.read_csv(list_path, sep=' ', header=None, names=['img_path', 'pid'])
        
        # Extract camera ID from the image path
        df['camid'] = df['img_path'].apply(lambda x: int(x.split('_')[2]))
        
        # Update image paths to include the directory path
        df['img_path'] = df['img_path'].apply(lambda x: osp.join(dir_path, x))
        
        # Adjust PID and camera ID
        df['pid'] += self.pid_begin
        df['camid'] -= 1
        
        # Add a trackid column with default value 1
        df['trackid'] = 1  
        
        # # Check if PID starts from 0 and increments with 1
        # pid_container = df['pid'].unique()
        # for idx, pid in enumerate(sorted(pid_container)):
        #     assert idx == pid, "See code comment for explanation"
        
        return df