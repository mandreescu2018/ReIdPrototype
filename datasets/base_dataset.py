import os
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(object):
    
    def get_imagedata_info(self, data):
        """
        get images information
        return:
            number of person identities
            length of the data
            number of cameras
            number of tracks (views)
        """

        pids, cams, tracks = set(), set(), set()
        for _, pid, camid, trackid in data:
            pids.add(pid)
            cams.add(camid)
            tracks.add(trackid)

        return len(pids), len(data), len(cams), len(tracks)
    
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


class ImageDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.data[index]
        img = self.read_image(img_path)
        img = self.transform(img)
        return img, pid, camid, trackid
    
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