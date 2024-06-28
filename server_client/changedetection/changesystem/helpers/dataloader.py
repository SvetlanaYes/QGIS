from torch.utils.data import DataLoader

import json
import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms.functional as TF

import sys
from .constants import PROJECT_CONFIGS
env_path = os.path.join('..', os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)



"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""

class CDDataset(data.Dataset):

    def __init__(self, root_dir, config):
        super(CDDataset, self).__init__()
        self.root_dir = root_dir
        self.config = config
        self.list_path = os.path.join(self.root_dir, self.config["image_names_dir"], self.config["split"]+'.txt')
        self.img_name_list = self.load_img_name_list(self.list_path)
        self.A_size = len(self.img_name_list)  # get the size of dataset A

    @staticmethod
    def load_img_name_list(dataset_path):
        #print(dataset_path)
        img_name_list = np.loadtxt(dataset_path, dtype=str)
        if img_name_list.ndim == 2:
            return img_name_list[:, 0]
        return np.array([str(img_name_list)])

    @staticmethod
    def load_image_label_list_from_npy(npy_path, img_name_list):
        cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
        return [cls_labels_dict[img_name] for img_name in img_name_list]

    def get_img_post_path(self, root_dir, img_name):
        return os.path.join(root_dir, self.config["split"], self.config["second_image_dir"], img_name)

    def get_img_path(self, root_dir, img_name):
        return os.path.join(root_dir, self.config["split"], self.config["first_image_dir"], img_name)

    def get_label_path(self, root_dir, img_name):
        return os.path.join(root_dir, self.config["split"], self.config["label_dir"], img_name)  # .replace('.jpg', label_suffix))

    @staticmethod
    def _prepare_images(imgs, labels):
        imgs = [TF.to_pil_image(img) for img in imgs]
        labels = [TF.to_pil_image(img) for img in labels]
        imgs = [TF.to_tensor(img) for img in imgs]
        labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                    for img in labels]

        imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                for img in imgs]
        
        return imgs, labels

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = self.get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = self.get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        L_path = self.get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        # TO DO .......
        if not os.path.exists(L_path):
            [img, img_B], _ = self._prepare_images([img, img_B], [])
            return {'name': name, self.config["first_image_dir"]: img, self.config["second_image_dir"]: img_B}

        label = np.array(Image.open(L_path).convert('L'), dtype=np.uint8)
        if self.config["label_transform"] == 'norm':
            label = label // 255

        [img, img_B], [label] = self._prepare_images([img, img_B], [label])
        return {'name': name, self.config["first_image_dir"]: img, self.config["second_image_dir"]: img_B, self.config["label_dir"]: label}

    def __len__(self):

        return self.A_size


def get_loader(data_name, batch_size=8, dataset='CDDataset'):
    absolute_path = os.path.abspath(PROJECT_CONFIGS)
    with open(absolute_path) as f:
        config = json.load(f)

    root_dir = config["datasets"][data_name]
    print(root_dir)
    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, config=config)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = False
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader


