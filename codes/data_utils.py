import yaml
import numpy as np

import torch
import torch.nn as nn

from torchsparse.nn import functional as F
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate
import torchsparse.nn as spnn

F.set_kmap_mode('hashmap')
F.set_downsample_mode('minkowski') 
conv_config = F.conv_config.get_default_conv_config()
F.conv_config.set_global_conv_config(conv_config)

class SimpleKittiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        list_files,
        config_file = './config/semantic-kitti.yaml'
    ):
        self.list_files = list_files
        
        LABEL_MAP = np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
                                  19, 19, 2, 19, 19, 3, 19, 4, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 5, 6, 7, 19, 19, 19, 19, 19, 19,
                                  19, 8, 19, 19, 19, 9, 19, 19, 19, 10, 11, 12, 13,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 14, 15, 16, 19, 19, 19, 19, 19,
                                  19, 19, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                                  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19])

        DATA = yaml.safe_load(open(config_file, 'r'))

        # get number of interest classes, and the label mappings
        class_strings = DATA["labels"]
        class_remap = DATA["learning_map"]
        class_inv_remap = DATA["learning_map_inv"]
        class_ignore = DATA["learning_ignore"]
        nr_classes = len(class_inv_remap)

        # make lookup table for mapping
        maxkey = max(class_remap.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        self.remap_lut[list(class_remap.keys())] = list(class_remap.values())
        
    def __len__(self):
        return len(self.list_files)
    def __getitem__(self, idx):
        fdict = self.list_files[idx]
        f_lidar = fdict['lidar']
        f_label = fdict['label']
        lidar = np.fromfile(f_lidar, dtype = np.float32).reshape((-1,4))
        label = np.fromfile(f_label, dtype = np.int32)
        
        lidar = lidar.reshape(-1, 4)
        label = self.remap_lut[(np.fromfile(fdict['label'], dtype = np.uint32) & 0xFFFF)]
        # Filter out ignored points
        # lidar = lidar[label != 0]
        # label = label[label != 0]
        # make torch.Tensor
        lidar = torch.Tensor(lidar)
        label = torch.Tensor(label).long()
        return lidar, label, fdict#, lidar_min

def collate_fn(list_x):
    list_lidar = []
    # list_lidar_min = []
    list_label = []
    list_index = []
    list_fdict = []
    for idx, (lidar, label, fdict) in enumerate(list_x):
        list_lidar.append(lidar)
        # list_lidar_min.append(lidar_min)
        list_label.append(label)
        list_index.append(torch.ones(len(lidar), 1) * idx)
        list_fdict.append(fdict)
    index = torch.cat(list_index, dim = 0).to(torch.int32)
    lidar = torch.cat(list_lidar, dim = 0)
    label = torch.cat(list_label, dim = 0)
    lidar = torch.cat([index, lidar], dim = -1)
    return lidar, label, list_fdict