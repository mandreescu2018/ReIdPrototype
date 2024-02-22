import os
import glob
import re
from torch.utils.data import Dataset
from.base_dataset import BaseDataset


class Market1501(BaseDataset):
    def __init__(self, cfg, verbose=True, pid_begin=0):
        self.root = cfg.DATASETS.ROOT_DIR
        self.dataset_dir = os.path.join(self.root, cfg.DATASETS.DIR)
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

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue # junk images are igmored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        
        return dataset