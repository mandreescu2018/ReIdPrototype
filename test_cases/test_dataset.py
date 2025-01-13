import sys
sys.path.append('.')

import unittest
from config import cfg
from datasets import make_dataloader

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('setupClass')
        cfg.DATASETS.NAMES = 'market1501'
        data = make_dataloader(cfg)
        cls.train_loader = data[0] 
        cls.test_loader = data[1] 
        cls.num_classes = data[2]
        cls.number_of_cameras = data[3]
        cls.number_of_camera_tracks = data[4]
        cls.query_num = data[5]
        
    # def setUp(self):
    #     self.train_loader, self.test_loader, self.num_classes, self.number_of_cameras, self.number_of_camera_tracks, self.query_num = make_dataloader(cfg)

    def test_number_of_classes(self):
        self.assertEqual(TestDataset.num_classes, 751)
    
    def test_number_of_cameras(self):
        self.assertEqual(TestDataset.number_of_cameras, 6)
    
    def test_number_of_camera_tracks(self):
        self.assertEqual(TestDataset.number_of_camera_tracks, 1)
    
    def test_query_num(self):
        self.assertEqual(TestDataset.query_num, 3368)   
    
    def test_train_loader(self):
        self.assertEqual(len(TestDataset.train_loader.dataset), 12936)
    
    def test_test_loader(self):
        self.assertEqual(len(TestDataset.test_loader.dataset), 19281)

if __name__ == '__main__':
    unittest.main()
    
