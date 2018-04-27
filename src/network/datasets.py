import os
import cv2
import time
import json
import torch
import random
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset 
from torchvision import transforms

class VoxelDataset(Dataset):
    """
    Instance vars:
        - self.data_base_dir (string with root data dir)
        - self.ids
    """

    def __init__(self, dset_type, data_base_dir):
        # Save input
        self.data_base_dir = data_base_dir

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

class ImageVoxelDataset:
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
        self.im_counts = {}
        for id_ in ids:
            id_ = str(id_)
            dir_ = os.path.join(self.data_base_dir, "input_ims", id_)
            files = [f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))]
            self.im_counts[id_] = len(files)
            self.ims += files

        # Create transform for image
        #   1) resize to appropriate dim
        #   2) crop to 224x224
        #   3) randomly flip horizontally
        #   3) rescale to 0->1 and reorder channels
        #   4) normalize to imagenet mean
        if "Redwood" in self.data_base_dir:
            resize = (237,237) #exactly 237x237
        else:
            resize = 347 #min dimension is 347
        self.transform_im = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize),
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
        data['im_name'] = im_name

        # Load image and process w/ transforms
        im_fp = os.path.join(self.data_base_dir, "input_ims", id_, im_name)
        img = cv2.imread(im_fp)
        data['im'] = self.transform_im(img)

        # Load voxel
        mat_fp = os.path.join(self.data_base_dir, "mat", "%s.mat" % id_)
        mat_data = scio.loadmat(mat_fp)
        data['voxel'] = mat_data["data"].astype(int)

        return data

class FusionDataset(Dataset):
    def __init__(self, src_dataset, target_dataset, src_subset_size=None, target_subset_size=None):
        self.src_dataset = src_dataset
        self.target_dataset = target_dataset

        # Subset if necessary
        if src_subset_size is None:
            self.src_inds = range(len(self.src_dataset))
        else:
            self.src_inds = random.sample(
                range(len(self.src_dataset)), 
                min(src_subset_size, len(self.src_dataset)))
        if target_subset_size is None:
            self.target_inds = range(len(self.target_dataset))
        else:
            self.target_inds = random.sample(
                range(len(self.target_dataset)), 
                min(target_subset_size, len(self.target_dataset)))
        return

    def __len__(self):
        return len(self.src_inds) + len(self.target_inds)

    def __getitem__(self, idx):
        if idx < len(self.src_inds):
            data = self.src_dataset[self.src_inds[idx]]
            data['domain'] = 'src'
        else:
            data = self.target_dataset[self.target_inds[idx - len(self.src_inds)]]
            data['domain'] = 'target'
        return data
