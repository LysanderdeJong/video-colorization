import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math
from argparse import ArgumentParser
from dataloader import VideoDataset, ImageDataset
import wandb
import numpy as np
import cv2


class ConvSame(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(ConvSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        N, C, H, W = x.shape
        H2 = math.ceil(H / self.S)
        W2 = math.ceil(W / self.S)
        Pr = (H2 - 1) * self.S + (self.F - 1) * self.D + 1 - H
        Pc = (W2 - 1) * self.S + (self.F - 1) * self.D + 1 - W
        x = nn.ZeroPad2d((Pr // 2, Pr - Pr // 2, Pc // 2, Pc - Pc // 2))(x)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        scale_factor=(2, 2),
        kernel_size=3,
        stride=1,
        dilation=1,
    ):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv = ConvSame(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        if self.scale_factor != (1, 1):
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
            )
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, source_in, source_out, ref_in, ref_out):
        super(Attention, self).__init__()
        self.source = nn.Conv2d(
            in_channels=source_in, out_channels=source_out, kernel_size=1
        )
        self.ref = nn.Conv2d(in_channels=ref_in, out_channels=ref_out, kernel_size=1)
        self.gate = nn.Conv2d(in_channels=ref_in, out_channels=ref_in, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, source_features, reference_features):
        batch_s, c_s, h_s, w_s = source_features.shape
        batch_r, c_r, h_r, w_r = source_features.shape

        source = (
            self.source(source_features).view(batch_s, -1, h_s * w_s).permute(0, 2, 1)
        )
        reference = self.ref(reference_features).view(batch_r, -1, h_r * w_r)
        gate = self.gate(reference_features).view(batch_r, -1, h_r * w_r)

        energy = torch.matmul(source, reference)
        attention = F.softmax((c_s**-0.5) * energy, dim=-1)

        out = torch.matmul(gate, attention.permute(0, 2, 1))
        out = out.view(batch_s, c_s, h_s, w_s)
        out = self.gamma * out + source_features
        return out


class Concat(nn.Module):
    def __init__(self, input1_c, input2_c, input3_c):
        super(Concat, self).__init__()

        self.upsample1 = ConvSame(input1_c * 2, input1_c)
        self.upsample2 = ConvSame(input2_c + input3_c, input1_c)
        self.batchnorm = nn.BatchNorm2d(input1_c)

    def forward(self, input1, input2, input3):
        input3 = F.interpolate(
            input3,
            size=(input2.shape[-2], input2.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        input2_input3_concat = torch.cat([input2, input3], dim=1)
        output_2 = F.relu(self.upsample2(input2_input3_concat))
        output_2 = self.batchnorm(output_2)

        output_2 = F.interpolate(
            output_2,
            size=(input1.shape[-2], input1.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        input1_c_output_2_concat = torch.cat([input1, output_2], dim=1)
        output_1 = F.relu(self.upsample1(input1_c_output_2_concat))
        return output_1


class ResNet_encoder(nn.Module):
    def __init__(
        self,
        model_type="resnet50",
        pretrained=True,
        fixed_extractor="partial",
        color_mode="gray",
    ):
        super(ResNet_encoder, self).__init__()

        if model_type == "resnet34":
            resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif model_type == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif model_type == "resnet101":
            resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif model_type == "resnet152":
            resnet = torchvision.models.resnet152(pretrained=pretrained)
        else:
            print(
                "model_type not reconized. Choose resnet34, resnet50, resnet101 or resnet152."
            )

        if color_mode == "gray":
            resnet.conv1.weight = nn.Parameter(
                resnet.conv1.weight.mean(dim=1).unsqueeze(1)
            )
        elif color_mode == "color":
            pass
        else:
            print("color_mode unknown. Choose gray or color.")

        self.resnet1 = torch.nn.Sequential(*(list(resnet.children())[:3]))
        self.resnet2 = torch.nn.Sequential(*(list(resnet.children())[3:5]))
        self.resnet3 = torch.nn.Sequential(*(list(resnet.children())[5:6]))
        self.resnet4 = torch.nn.Sequential(*(list(resnet.children())[6:7]))
        self.resnet5 = torch.nn.Sequential(*(list(resnet.children())[7:8]))

        if fixed_extractor == "partial" or fixed_extractor == "full":
            for param in self.resnet1.parameters():
                param.requires_grad = False
            for param in self.resnet2.parameters():
                param.requires_grad = False
            for param in self.resnet3.parameters():
                param.requires_grad = False
        if fixed_extractor == "full":
            for param in self.resnet4.parameters():
                param.requires_grad = False
            for param in self.resnet5.parameters():
                param.requires_grad = False

    def forward(self, x):
        output1 = self.resnet1(x)  # (64, H//2, W//2)
        output2 = self.resnet2(output1)  # (265, H//4, W//4)
        output3 = self.resnet3(output2)  # (512, H//8, W//8)
        output4 = self.resnet4(output3)  # (1024, H//16, W//16)
        output5 = self.resnet5(output4)  # (2048, H//32, W//32)
        return output3, output4, output5


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            ConvSame(1024, 512),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            Upsample(512, 265, scale_factor=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(265),
            ConvSame(265, 128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            ConvSame(128, 64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            Upsample(64, 32, scale_factor=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            ConvSame(32, 16),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            ConvSame(16, 8),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            ConvSame(8, 2),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class ColorNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ColorNet, self).__init__()
        self.hparams = hparams

        self.encoder = ResNet_encoder(
            model_type="resnet50",
            pretrained=True,
            fixed_extractor="full",
            color_mode="color",
        )
        self.concat = Concat(512, 1024, 2048)
        self.sourcerefattention = Attention(512, 128, 512, 128)
        self.selfattention1 = Attention(512, 128, 512, 128)
        self.decoder = Decoder()
        self.tanh = nn.Tanh()

        self.f_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, img_gray, ref_color=None):
        img_gray_1, img_gray_2, img_gray_3 = self.encoder(img_gray)
        x1 = self.concat(img_gray_1, img_gray_2, img_gray_3)

        if ref_color is not None:
            ref_color_1, ref_color_2, ref_color_3 = self.encoder(ref_color)
            ref_color_features = self.concat(ref_color_1, ref_color_2, ref_color_3)
            x1 = self.sourcerefattention(x1, ref_color_features)

        self_attention = self.selfattention1(x1, x1)
        out = torch.cat([self_attention, img_gray_1], dim=1)
        out = self.decoder(out)
        out = self.tanh(out)
        return out

    def prepare_data(self):
        self.train_dataset = VideoDataset(
            self.hparams.train_dir, transform=self.hparams.augmentation
        )
        self.test_dataset = ImageDataset(self.hparams.test_dir)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.hparams.patience
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_indx):
        data, target = batch

        output = self.forward(data[0], data[1])
        output = F.interpolate(
            output,
            size=(data[0].shape[-2], data[0].shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        loss = self.f_loss(output, target)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        wandb_logs = {"loss": avg_loss, "epoch": self.current_epoch}
        return {"avg_loss": avg_loss, "log": wandb_logs}

    def validation_step(self, batch, batch_indx):
        data, target = batch

        output = self.forward(data[0], data[1])
        output = F.interpolate(
            output,
            size=(data[0].shape[-2], data[0].shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        val_loss = self.f_loss(output, target)

        batch_size = data[0].shape[0]
        random_index = np.random.randint(0, batch_size)

        inv_norm = torchvision.transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

        gray = inv_norm(data[0][random_index]).cpu()
        ref = inv_norm(data[1][random_index]).cpu()
        ab_color = target[random_index].cpu()
        pred = output[random_index].cpu()

        return {
            "val_loss": val_loss,
            "gray": gray,
            "ref": ref,
            "target": ab_color,
            "pred": pred,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        gray = [x["gray"] for x in outputs][0].permute(1, 2, 0)
        gray = gray[:, :, 0].unsqueeze(-1) * 100
        target = [x["target"] for x in outputs][0].permute(1, 2, 0) * 110
        pred = [x["pred"] for x in outputs][0].permute(1, 2, 0) * 110

        target = torch.cat([gray, target], dim=-1)
        pred = torch.cat([gray, pred], dim=-1)

        target = np.array(target)
        pred = np.array(pred)

        target = cv2.cvtColor(target, cv2.COLOR_LAB2RGB)
        target = (target * 255).astype(np.uint8)

        pred = cv2.cvtColor(pred, cv2.COLOR_LAB2RGB)
        pred = (pred * 255).astype(np.uint8)

        gray = gray.squeeze(-1)
        gray = np.array(gray)
        gray = (gray * (255 / 100)).astype(np.uint8)

        ref = [x["ref"] for x in outputs][0].permute(1, 2, 0)
        ref = np.array(ref)
        ref = (ref * 255).astype(np.uint8)

        self.logger.experiment.log(
            {
                "input": [wandb.Image(gray, caption="Grayscale Input")],
                "predicted": [wandb.Image(pred, caption="Predicted Color")],
                "ground_truth": [wandb.Image(target, caption="Ground Truth")],
                "reference": [wandb.Image(ref, caption="Reference Color")],
            }
        )

        wandb_logs = {"val_loss": avg_loss, "epoch": self.current_epoch}

        return {"avg_val_loss": avg_loss, "log": wandb_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--train_dir", default=None, type=str)
        parser.add_argument("--test_dir", default=None, type=str)

        parser.add_argument("--augmentation", default=True, type=bool)

        parser.add_argument("--num_workers", default=6, type=int)

        parser.add_argument("--patience", default=10, type=int)

        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        return parser
