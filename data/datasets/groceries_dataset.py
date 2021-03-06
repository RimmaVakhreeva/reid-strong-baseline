import os.path as osp
import pandas as pd

from .bases import BaseImageDataset


class GroceriesDataset(BaseImageDataset):
    dataset_dir = 'groceries_dataset'

    def __init__(self, root, verbose=True, **kwargs):
        super(GroceriesDataset, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, 'train.csv')
        query = self._process_dir(self.query_dir, 'query.csv')
        gallery = self._process_dir(self.gallery_dir, 'gallery.csv')

        if verbose:
            print('=> GroceriesDataset loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, csv_path):
        train_df = pd.read_csv(osp.join(self.dataset_dir, csv_path))
        dataset = []
        for index, row in train_df.iterrows():
            dataset.append((osp.join(dir_path, row['name']), row['class'], -1))
        return dataset
