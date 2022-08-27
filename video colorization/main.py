import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from network import ColorNet

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    if hparams.supress_logs:
        import logging

        logger = logging.getLogger("wandb")
        logger.setLevel(logging.ERROR)

    colornet = model.ColorNet()
    wandb_logger = WandbLogger(project="video-colorization", tags=["colornet"])
    wandb_logger.watch(colornet, log_freq=hparams.log_frequency)

    early_stopping = EarlyStopping("val_loss", patience=hparams.patience)
    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/checkpoint_{epoch:02d}-{val_loss:.2f}"
    )
    trainer = pl.Trainer(
        max_epochs=hparams.epoch,
        gpus=hparams.gpus,
        logger=wandb_logger,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(colornet)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    parser = ColorNet.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    main(hyperparams)
