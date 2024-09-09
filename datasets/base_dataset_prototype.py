class BaseDataset_prototype(object):
    
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
        # pids, cams, tracks = set(), set(), set()
        # for _, pid, camid, trackid in data:
        #     pids.add(pid)
        #     cams.add(camid)
        #     tracks.add(trackid)

        # return len(pids), len(data), len(cams), len(tracks)
    
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
