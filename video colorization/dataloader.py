import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transform
import numpy as np
import cv2
import albumentations as A


class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.video_dirs = os.listdir(self.video_dir)
        self.transform = transform
        self.data_len = 0
        self.video_dir_dict = {}
        for video in self.video_dirs:
            tmp = self.data_len
            self.data_len += len(os.listdir(os.path.join(self.video_dir, video)))
            self.video_dir_dict[video] = (tmp, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        for video, index_range in self.video_dir_dict.items():
            if idx >= index_range[0] and idx < index_range[1]:
                img_index = np.clip(idx - index_range[0], 0, None)
                img_path = os.path.join(self.video_dir, video, os.listdir(os.path.join(self.video_dir, video))[img_index])

                ref_index = np.random.randint(0, index_range[1] - index_range[0])
                ref_path = os.path.join(self.video_dir, video, os.listdir(os.path.join(self.video_dir, video))[ref_index])

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ref = cv2.imread(ref_path)
        ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_spatial_aug = img_spatial_augmentation(img_rgb)
            img_pixel_aug = img_pixel_augmentation(img_spatial_aug)
            ref_rgb_aug = ref_spatial_augmentation(ref_rgb)

            img_spatial_aug = (np.array(img_spatial_aug) / 255.).astype(np.float32)
            img_pixel_aug = (np.array(img_pixel_aug) / 255.).astype(np.float32)
            ref_rgb = (np.array(ref_rgb_aug) / 255.).astype(np.float32)

            img_l = cv2.cvtColor(img_pixel_aug, cv2.COLOR_RGB2LAB)[:, :, 0] / 100.
            img_ab = cv2.cvtColor(img_spatial_aug, cv2.COLOR_RGB2LAB)[:, :, 1:] / 110.
        else:
            ref_rgb = resize(ref_rgb, size=(224, 224))
            ref_rgb = (np.array(ref_rgb) / 255.).astype(np.float32)

            img_rgb = resize(img_rgb, size=(224, 224))
            img_rgb = (np.array(img_rgb) / 255.).astype(np.float32)

            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            img_l = img_lab[:, :, 0] / 100.
            img_ab = img_lab[:, :, 1:] / 110.

        img_ref = torch.from_numpy(ref_rgb).float().permute(2, 0, 1)
        img_l = torch.from_numpy(img_l).float().unsqueeze(0)
        img_ab = torch.from_numpy(img_ab).float().permute(2, 0, 1)

        img_l = torch.cat([img_l, img_l, img_l], dim=0)

        norm = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_ref = norm(img_ref)
        img_l = norm(img_l)
        return ([img_l, img_ref], img_ab)


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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_spatial_aug = img_spatial_augmentation(img_rgb)
            img_pixel_aug = img_pixel_augmentation(img_spatial_aug)
            img_rgb_aug = ref_spatial_augmentation(img_rgb)

            img_spatial_aug = (np.array(img_spatial_aug) / 255.).astype(np.float32)
            img_pixel_aug = (np.array(img_pixel_aug) / 255.).astype(np.float32)
            img_ref = (np.array(img_rgb_aug) / 255.).astype(np.float32)

            img_l = cv2.cvtColor(img_pixel_aug, cv2.COLOR_RGB2LAB)[:, :, 0] / 100.
            img_ab = cv2.cvtColor(img_spatial_aug, cv2.COLOR_RGB2LAB)[:, :, 1:] / 110.
        else:
            img_rgb = resize(img_rgb, size=(224, 224))
            img_ref = (np.array(img_rgb) / 255.).astype(np.float32)

            img_lab = cv2.cvtColor(img_ref, cv2.COLOR_RGB2LAB)
            img_l = img_lab[:, :, 0] / 100.
            img_ab = img_lab[:, :, 1:] / 110.

        img_ref = torch.from_numpy(img_ref).float().permute(2, 0, 1)
        img_l = torch.from_numpy(img_l).float().unsqueeze(0)
        img_ab = torch.from_numpy(img_ab).float().permute(2, 0, 1)

        img_l = torch.cat([img_l, img_l, img_l], dim=0)

        norm = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_ref = norm(img_ref)
        img_l = norm(img_l)
        return ([img_l, img_ref], img_ab)


def img_spatial_augmentation(img, p=1.0, size=(224, 224)):
    pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.75),
        A.RandomSizedCrop(min_max_height=(160, 256), height=size[0], width=size[1], p=1.0)
    ], p=p)

    augmented = pipeline(image=img)
    return augmented['image']


def img_pixel_augmentation(img, p=0.5):
    pipeline = A.Compose([A.RGBShift(),
                          A.OneOf([
                              A.IAAAdditiveGaussianNoise(),
                              A.GaussNoise()
                          ], p=0.2),
                         ], p=p)

    augmented = pipeline(image=img)
    return augmented['image']


def ref_spatial_augmentation(img, p=1.0, size=(224, 224)):
    pipeline = A.Compose([
        A.ShiftScaleRotate(p=0.7),
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_spatial_aug = img_spatial_augmentation(img_rgb)
            img_pixel_aug = img_pixel_augmentation(img_spatial_aug)
            img_rgb_aug = ref_spatial_augmentation(img_rgb)

            img_spatial_aug = (np.array(img_spatial_aug) / 255.).astype(np.float32)
            img_pixel_aug = (np.array(img_pixel_aug) / 255.).astype(np.float32)
            img_ref = (np.array(img_rgb_aug) / 255.).astype(np.float32)

            img_l = cv2.cvtColor(img_pixel_aug, cv2.COLOR_RGB2LAB)[:, :, 0] / 100.
            img_ab = cv2.cvtColor(img_spatial_aug, cv2.COLOR_RGB2LAB)[:, :, 1:] / 110.
        else:
            img_rgb = resize(img_rgb, size=(224, 224))
            img_ref = (np.array(img_rgb) / 255.).astype(np.float32)

            img_lab = cv2.cvtColor(img_ref, cv2.COLOR_RGB2LAB)
            img_l = img_lab[:, :, 0] / 100.
            img_ab = img_lab[:, :, 1:] / 110.

        img_ref = torch.from_numpy(img_ref).float().permute(2, 0, 1)
        img_l = torch.from_numpy(img_l).float().unsqueeze(0)
        img_ab = torch.from_numpy(img_ab).float().permute(2, 0, 1)

        img_l = torch.cat([img_l, img_l, img_l], dim=0)

        norm = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_ref = norm(img_ref)
        img_l = norm(img_l)
        return ([img_l, img_ref], img_ab)