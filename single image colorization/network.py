import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math
import cv2
import os
import numpy as np
import random
import wandb
from argparse import ArgumentParser
from dataloader import ImageDataset, Imagenet_Subset
from predict_images import predict_images

from modules.cielab import CIELAB, DEFAULT_CIELAB
from modules.annealed_mean_decode_q import AnnealedMeanDecodeQ
from modules.soft_encode_ab import SoftEncodeAB
from modules.get_class_weights import GetClassWeights
from modules.rebalance_loss import RebalanceLoss
from modules.cross_entropy_loss_2d import CrossEntropyLoss2d
from modules.deeplab_v3_plus import DeepLabV3Plus
from modules.vgg_segmentation_network import VGGSegmentationNetwork


class ConvSame(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(ConvSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        N, C, H, W = x.shape
        H2 = math.ceil(H / self.S)
        W2 = math.ceil(W / self.S)
        Pr = (H2 - 1) * self.S + (self.F - 1) * self.D + 1 - H
        Pc = (W2 - 1) * self.S + (self.F - 1) * self.D + 1 - W
        x = nn.ZeroPad2d((Pr//2, Pr - Pr//2, Pc//2, Pc - Pc//2))(x)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=(2, 2), kernel_size=3, stride=1, dilation=1):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv = ConvSame(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        if self.scale_factor != (1, 1):
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, source_in, source_out, ref_in, ref_out):
        super(Attention, self).__init__()
        self.source = nn.Conv2d(in_channels=source_in, out_channels=source_out, kernel_size=1)
        self.ref = nn.Conv2d(in_channels=ref_in, out_channels=ref_out, kernel_size=1)
        self.gate = nn.Conv2d(in_channels=ref_in, out_channels=ref_in, kernel_size=1)

    def forward(self, source_features, reference_features):
        batch_s, c_s, h_s, w_s = source_features.shape
        batch_r, c_r, h_r, w_r = source_features.shape

        source = self.source(source_features).view(batch_s, -1, h_s * w_s).permute(0, 2, 1)
        reference = self.ref(reference_features).view(batch_r, -1, h_r * w_r)
        gate = self.gate(reference_features).view(batch_r, -1, h_r * w_r)

        energy = torch.matmul(source, reference)
        attention = F.softmax((c_s ** -.5) * energy, dim=-1)

        out = torch.matmul(gate, attention.permute(0, 2, 1))
        out = out.view(batch_s, c_s, h_s, w_s)
        return out


class ResNet_encoder(nn.Module):
    def __init__(self, pretrained=True, fixed_extractor=False, mode="gray"):
        super(ResNet_encoder, self).__init__()

        if pretrained:
            resnet = torchvision.models.resnet101(pretrained=True)
        else:
            resnet = torchvision.models.resnet101(pretrained=False)

        if mode == "gray":
            resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1).unsqueeze(1))
        elif mode == "color":
            pass
        else:
            print("Mode unknown. Choose gray or color.")

        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-4]))

        if fixed_extractor:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        # (512, H//8, W//8)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
                        Upsample(512, 512, scale_factor=(1, 1), dilation=2),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(512),
                        Upsample(512, 512, scale_factor=(1, 1), dilation=2),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(512),
                        Upsample(512, 512, scale_factor=(1, 1), dilation=2),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(512),
                        Upsample(512, 512, scale_factor=(1, 1)),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(512),
                        Upsample(512, 512, scale_factor=(1, 1)),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(512),
                        Upsample(512, 512, scale_factor=(1, 1)),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(512),
                        Upsample(512, 265, scale_factor=(2, 2), kernel_size=4),
                        nn.LeakyReLU(),
                        Upsample(265, 265, scale_factor=(1, 1)),
                        nn.LeakyReLU(),
                        Upsample(265, 265, scale_factor=(1, 1)),
                        nn.LeakyReLU(),
                        Upsample(265, 313, scale_factor=(1, 1), kernel_size=1)
                        )

    def forward(self, x):
        x = self.decoder(x)
        # (313, H//4, W//4)
        return x


class ColorNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ColorNet, self).__init__()
        self.hparams = hparams

        self.network = DeepLabV3Plus(313)
        #self.network.init_from_tensorflow('pretrained/xception_65_coco_pretrained/model.ckpt')

        # self.network = VGGSegmentationNetwork(313)

        self.encode_ab = SoftEncodeAB(DEFAULT_CIELAB, device=self.device)

        self.decode_q = AnnealedMeanDecodeQ(DEFAULT_CIELAB,
                                            T=0.38, device=self.device)

        self.get_class_weights = GetClassWeights(DEFAULT_CIELAB, device=self.device)

        self.rebalance_loss = RebalanceLoss.apply

        self.f_loss = CrossEntropyLoss2d()

    def forward(self, img_l):
        x = self.network(img_l)
        return x

    def prepare_data(self):
        self.train_dataset = Imagenet_Subset(self.hparams.train_dir, check_percent=0.2,
                                             transform=self.hparams.augmentation)
        self.test_dataset = ImageDataset(self.hparams.test_dir)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, shuffle=True,
                                  batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers,
                                  drop_last=True, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        test_loader = DataLoader(self.test_dataset, shuffle=False,
                                 batch_size=self.hparams.batch_size,
                                 num_workers=self.hparams.num_workers,
                                 drop_last=True, pin_memory=True)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.hparams.patience, verbose=True, factor=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_indx):
        # data = full res grayscale, target = quarter res
        data, target = batch

        output = self.forward(data)
        target = self.encode_ab(target)
        color_weights = self.get_class_weights(target)
        output = self.rebalance_loss(output, color_weights)

        loss = self.f_loss(output, target)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        wandb_logs = {"loss": avg_loss, 'epoch': self.current_epoch}
        return {"avg_loss": avg_loss, "log": wandb_logs}

    def validation_step(self, batch, batch_indx):
        data, target = batch

        output = self.forward(data)
        target = self.encode_ab(target)
        # color_weights = self.get_class_weights(target)
        # output = self.rebalance_loss(output, color_weights)

        val_loss = self.f_loss(output, target)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        img_dir = self.hparams.test_dir
        img_path = random.choice(os.listdir(img_dir))
        img_path = os.path.join(img_dir, img_path)

        img = (cv2.imread(img_path) / 255.).astype(np.float32)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        gt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        img_l = img[:, :, 0] / 100

        img_l_t = torch.from_numpy(img_l).float()

        img_l_t = torch.unsqueeze(img_l_t, 0)

        img_l = torch.cat([img_l_t, img_l_t, img_l_t], dim=0)

        norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_l = norm(img_l)
        #img_l = img_l_t *100
        #img_l = img_l - 50

        with torch.cuda.device(1):
            with torch.no_grad():
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

        self.logger.experiment.log({
            "input": [wandb.Image(gray, caption="Grayscale Input")],
            "predicted": [wandb.Image(rgb, caption="Predicted Color")],
            "ground_truth": [wandb.Image(gt, caption="Ground Truth")]
        })

        # predict_images(self, cuda_device=1)

        wandb_logs = {"val_loss": avg_loss, 'epoch': self.current_epoch}
        return {"avg_val_loss": avg_loss, "log": wandb_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--train_dir', default=None, type=str)
        parser.add_argument('--test_dir', default=None, type=str)

        parser.add_argument('--augmentation', default=True, type=bool)

        parser.add_argument('--num_workers', default=6, type=int)

        parser.add_argument('--patience', default=10, type=int)

        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        return parser