import os
import cv2
import time
import json
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset 
from torchvision import transforms

class ShapeNetVoxelDataset(Dataset):
    """
    Instance vars:
        - self.data_base_dir (string with root data dir)
        - self.transform
        - self.ids
    """

    def __init__(self, dset_type, data_base_dir, transform=None):
        # Save input
        self.data_base_dir = data_base_dir
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
        mat_fp = os.path.join(self.data_base_dir, "mat", "%s.mat" % id_)
        mat_data = scio.loadmat(mat_fp)
        data = {}
        data["id"] = id_
        data["data"] = mat_data["data"].astype(int)
        return data

class RenderingVoxelDataset:
    def __init__(self, dset_type, data_base_dir):
        self.data_base_dir = data_base_dir

        # Read train/val/test split json
        split_json_fp = os.path.join(self.data_base_dir, "json", "split.json")
        split_json_f = open(split_json_fp, 'r')
        split_json = json.loads(split_json_f.readlines()[0])
        split_json_f.close()

        # Extract list of ids
        assert(dset_type in split_json)
        ids = split_json[dset_type]

        # Get list of images
        self.ims = []
        for id_ in ids:
            dir_ = os.path.join(self.data_base_dir, "renderings", id_)
            files = [f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))]
            self.ims += files

        # Create transform for im renderings
        #   1) resize so min dimension is 224
        #   2) crop to 224x224
        #   3) randomly flip horizontally
        #   3) rescale to 0->1 and reorder channels
        #   4) normalize to imagenet mean
        self.transform_im = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(247),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):

        # Get im name and corresponding model id
        im_name = self.ims[idx]
        id_ = im_name.split("_")[1]
        data = {}

        # Load image and process w/ transforms
        im_fp = os.path.join(self.data_base_dir, "renderings", id_, im_name)
        img = cv2.imread(im_fp)
        data['im'] = self.transform_im(img)

        # Load voxel
        mat_fp = os.path.join(self.data_base_dir, "mat", "%s.mat" % id_)
        mat_data = scio.loadmat(mat_fp)
        data['voxel'] = mat_data["data"].astype(int)

        return data
