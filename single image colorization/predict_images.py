import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
from skimage import io
from argparse import ArgumentParser
from tqdm.notebook import tqdm


def predict_images(self, cuda_device=0):
    imagenet = "D:/Video Colorization/Datasets/Imagenet/CLS-LOC/test"
    coco = "D:/Video Colorization/Datasets/COCO/val2014"
    places = "D:/Video Colorization/Datasets/Places/val_large"

    save_root = "D:/Video Colorization/Notebook/video/single image colorization/results"

    saves = ["imagenet", "places", "coco"]

    datasets = [imagenet, places, coco]

    with torch.cuda.device(cuda_device):
        with torch.no_grad():
            for img_dir in tqdm(range(len(datasets))):
                for img_index in tqdm(range(len(os.listdir(datasets[img_dir])))):
                    img_path = os.path.join(datasets[img_dir], os.listdir(os.path.join(datasets[img_dir]))[img_index])

                    img = (cv2.imread(img_path) / 255.).astype(np.float32)
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

                    img_l = img[:, :, 0] / 100

                    img_l_t = torch.from_numpy(img_l).float()

                    img_l_t = torch.unsqueeze(img_l_t, 0)

                    img_l = torch.cat([img_l_t, img_l_t, img_l_t], dim=0)

                    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                    img_l = norm(img_l)

                    img_l = torch.unsqueeze(img_l, 0).cuda()
                    output = self.forward(img_l)

                    ab = self.decode_q(output)
                    ab = F.interpolate(ab, size=(img_l.shape[-2], img_l.shape[-1]), mode="bilinear", align_corners=False)

                    img_l_t = img_l_t * 100
                    img_l_t = torch.unsqueeze(img_l_t, 0)
                    img_l_t = img_l_t.cuda()

                    lab = torch.cat([img_l_t, ab], dim=1)
                    lab = lab.squeeze(0)
                    lab = lab.cpu()
                    lab = np.array(lab.permute(1, 2, 0))

                    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    rgb = (rgb * 255).astype(np.uint8)

                    save_path = os.path.join(save_root, saves[img_dir], os.listdir(os.path.join(datasets[img_dir]))[img_index])
                    io.imsave(save_path, rgb, quality=90)