import network as model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser


def main(hparams):
    pl.seed_everything(42)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)

    wandb_logger = WandbLogger(project='video-colorization', tags=["colornet"],
                               name='SLURM', log_model=True)
    hparams.logger = wandb_logger
    lr_logger = LearningRateLogger()

    colornet = model.ColorNet(hparams)
    trainer = pl.Trainer.from_argparse_args(hparams, checkpoint_callback=checkpoint_callback,
                                            callbacks=[lr_logger])
    trainer.fit(colornet)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser = model.ColorNet.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)