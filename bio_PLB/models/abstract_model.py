import os

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid, save_image
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate
from lightning_fabric.utilities.data import AttributeDict

from bio_PLB.tools import get_git_revision_short_hash


class AbstractModel(pl.LightningModule):
    def __init__(self, args_dict):
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

    def configure_optimizers(self):
        # instantiate the optimizer, passing model parameters
        optimizer = instantiate(
            self.hparams.args_dict.optimizer,
            params=self.parameters()
        )

        # instantiate the scheduler, passing the optimizer
        if self.hparams.args_dict.get("scheduler"):
            scheduler = instantiate(
                self.hparams.args_dict.scheduler,
                optimizer=optimizer
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # or "epoch"
                },
            }

        return { "optimizer": optimizer }


    def process_batch_supervised(self, batch):
        """get predictions, losses and mean errors (MAE)"""
        raise NotImplementedError("This method should be implemented in the subclass. It should return preds, losses and metrics dictionaries.")

    def log_all(self, losses, metrics, prefix=''):
        for k, v in losses.items():
            self.log(f'{prefix}{k}_loss', v.item() if isinstance(v, torch.Tensor) else v)

        for k, v in metrics.items():
            self.log(f'{prefix}{k}', v.item() if isinstance(v, torch.Tensor) else v)


    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='train_')

        return losses['final']

    def validation_step(self, batch, batch_idx):
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='val_')

    def test_step(self, batch, batch_idx):
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='test_')