import os
import cv2
import json
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset 

class ShapeNetVoxelDataset(Dataset):
    """
    Instance vars:
        - self.data_base_dir (string with root data dir)
        - self.transform
        - self.ids
    """

    def __init__(self, dset_type, data_base_dir, obj_class, transform=None):
        # Save input
        self.data_base_dir = data_base_dir
        self.obj_class = obj_class
        self.transform = transform

        # Read train/val/test split json
        split_json_fp = os.path.join(self.data_base_dir, "json", "split.json")
        split_json_f = open(split_json_fp, 'r')
        split_json = json.loads(split_json_f.readlines()[0])
        split_json_f.close()

        # Extract list of ids
        assert(dset_type in split_json)
        self.ids = split_json[dset_type]
        return

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        mat_fp = os.path.join(self.data_base_dir, obj_class, "mat", "%s.mat" % id_)
        mat_data = scio.loadmat(mat_fp)
        return mat_data['data']
