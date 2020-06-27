import torch
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transform
import numpy as np
import albumentations as A


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, check_percent=1.0):
        self.root = img_dir
        self.transform = transform
        self.images = os.listdir(self.root)
        self.data_len = len(self.images)

        if os.path.isdir(os.path.join(self.root, self.images[0])):
            img_list = []
            for img_dir in self.images:
                file_names = os.listdir(os.path.join(self.root, img_dir))
                for img in range(int(len(file_names) * check_percent)):
                    path = os.path.join(img_dir, file_names[img])
                    img_list.append(path)
            self.images = img_list
            self.data_len = len(self.images)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = img_aug(img)

        img_big = resize(img, size=(224, 224))
        img_small = resize(img, size=(56, 56))

        img_big = (np.array(img_big) / 255.).astype(np.float32)
        img_small = (np.array(img_small) / 255.).astype(np.float32)

        img_lab = cv2.cvtColor(img_big, cv2.COLOR_RGB2LAB)
        img_l = img_lab[:, :, 0] / 100.
        img_ab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)[:, :, 1:]

        img_ab = torch.from_numpy(img_ab).float().permute(2, 0, 1)
        img_l = torch.from_numpy(img_l).float()
        img_l = img_l.unsqueeze(0)

        img_l = torch.cat([img_l, img_l, img_l], dim=0)

        norm = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_l = norm(img_l)
        #img_l *= 100
        #img_l = img_l - 50

        return (img_l, img_ab)


def img_aug(img, p=1.0, size=(224, 224)):
    pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomSizedCrop(min_max_height=(160, 256), height=size[0], width=size[1], p=1.0)
    ], p=p)

    augmented = pipeline(image=img)
    return augmented['image']


def resize(img, size=(224, 224)):
    pipeline = A.Compose([
        A.Resize(size[0], size[1])
    ], p=1.0)

    resized = pipeline(image=img)
    return resized['image']


class Imagenet_Subset(Dataset):
    def __init__(self, img_dir, check_percent=1.0, transform=None):
        self.root = img_dir
        self.transform = transform
        self.images = os.listdir(self.root)
        self.data_len = len(self.images)

        if os.path.isdir(os.path.join(self.root, self.images[0])):
            img_list = []
            for img_dir in self.images:
                file_names = os.listdir(os.path.join(self.root, img_dir))
                for img in range(int(len(file_names) * check_percent)):
                    path = os.path.join(img_dir, file_names[img])
                    img_list.append(path)
            self.images = img_list
            self.data_len = len(self.images)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = img_aug(img)

        img_big = resize(img, size=(224, 224))
        img_small = resize(img, size=(56, 56))

        img_big = (np.array(img_big) / 255.).astype(np.float32)
        img_small = (np.array(img_small) / 255.).astype(np.float32)

        img_lab = cv2.cvtColor(img_big, cv2.COLOR_RGB2LAB)
        img_l = img_lab[:, :, 0] / 100.
        img_ab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)[:, :, 1:]

        img_ab = torch.from_numpy(img_ab).float().permute(2, 0, 1)
        img_l = torch.from_numpy(img_l).float()
        img_l = img_l.unsqueeze(0)

        img_l = torch.cat([img_l, img_l, img_l], dim=0)

        norm = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_l = norm(img_l)
        #img_l *= 100
        #img_l = img_l - 50

        return (img_l, img_ab)