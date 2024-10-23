import os.path as osp

class BaseDataset(object):
    
    def __init__(self) -> None:
        self._dataset_dir = None
        self._train_dir = None
        self._query_dir = None
        self._gallery_dir = None

    def get_imagedata_info(self, dataframe):
        """
        get images information
        return:
            number of person identities
            length of the data
            number of cameras
            number of tracks (views)
        """
        return dataframe['pid'].nunique(), \
                len(dataframe), \
                dataframe['camid'].nunique(), \
                dataframe['trackid'].nunique()
    
    @property
    def train_dir(self):
        return self._train_dir
    
    @train_dir.setter
    def train_dir(self, train_dir):
        if not osp.exists(train_dir):
            raise RuntimeError("'{}' is not available".format(train_dir))
        self._train_dir = train_dir

    @property
    def query_dir(self):
        return self._query_dir
    
    @query_dir.setter
    def query_dir(self, query_dir):
        if not osp.exists(query_dir):
            raise RuntimeError("'{}' is not available".format(query_dir))
        self._query_dir = query_dir

    @property
    def gallery_dir(self):
        return self._gallery_dir
    
    @gallery_dir.setter
    def gallery_dir(self, gallery_dir):
        if not osp.exists(gallery_dir):
            raise RuntimeError("'{}' is not available".format(gallery_dir))
        self._gallery_dir = gallery_dir

    @property
    def dataset_dir(self):
        return self._dataset_dir
    
    @dataset_dir.setter
    def dataset_dir(self, dataset_dir):
        if not osp.exists(dataset_dir):
            raise RuntimeError("'{}' is not available".format(dataset_dir))
        self._dataset_dir = dataset_dir

    
    def load_data_statistics(self):
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


    def print_dataset_statistics(self):

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print(f"  train    | {self.num_train_pids:5d} | {self.num_train_imgs:8d} | {self.num_train_cams:9d}")
        print(f"  query    | {self.num_query_pids:5d} | {self.num_query_imgs:8d} | {self.num_query_cams:9d}")
        print(f"  gallery  | {self.num_gallery_pids:5d} | {self.num_gallery_imgs:8d} | {self.num_gallery_cams:9d}")
        print("  ----------------------------------------")
