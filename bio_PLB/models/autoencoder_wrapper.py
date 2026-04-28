import os

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid, save_image
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate
from lightning_fabric.utilities.data import AttributeDict

from bio_PLB.tools import get_git_revision_short_hash


class AutoencoderWrapper(pl.LightningModule):
    def __init__(self, args_dict):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.generator_a = instantiate(args_dict.generator.model)
        self.generator_b = instantiate(args_dict.generator.model)
        init_weights(self.generator_a, args_dict.generator.weight_init)
        init_weights(self.generator_b, args_dict.generator.weight_init)
        self.loss = instantiate(args_dict.loss)

        self.masking = instantiate(args_dict.masking)

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

    def configure_optimizers(self):
        # instantiate the optimizer, passing model parameters
        optimizer = instantiate(
            self.hparams.args_dict.generator.optimizer,
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


    def forward(self, x):
        return #TODO

    def process_batch_supervised(self, batch):
        """get predictions, losses and mean errors (MAE)"""
        # execute the logic from uvcgan/cgan/autoencoder.py set_input & froward functions

        preds = AttributeDict()

        #image_types = ['real_a', 'reco_a', 'real_b', 'reco_b', 'masked_a', 'masked_b']
        #for image_type in image_types:
        #    preds[image_type] = None

        preds.real_a = batch[0].to(self.device)
        preds.real_b = batch[1].to(self.device)

        preds.masked_a = self.masking(preds.real_a)
        preds.masked_b = self.masking(preds.real_b)

        preds.reco_a = self.generator_a(preds.masked_a)
        preds.reco_b = self.generator_b(preds.masked_b)

        loss_a = self.loss(preds.real_b, preds.reco_b)
        loss_b = self.loss(preds.masked_a, preds.reco_a)


        losses  = { 'loss_a': loss_a,
                    'loss_b': loss_b,
                    'final': loss_a + loss_b
                  }

        metrics = { }

        return preds, losses, metrics


    def log_all(self, losses, metrics, prefix=''):
        for k, v in losses.items():
            self.log(f'{prefix}{k}_loss', v.item() if isinstance(v, torch.Tensor) else v)

        for k, v in metrics.items():
            self.log(f'{prefix}{k}', v.item() if isinstance(v, torch.Tensor) else v)

    def log_preds(self, preds, outdir):
        # make a new subdirectory in outdir - if not already - with get_git_revision_short_hash()
        subdir = os.path.join(outdir, get_git_revision_short_hash())
        os.makedirs(subdir, exist_ok=True)
        # for preds dictionary, save the batch of real_a, real_b, masked_a, masked_b, reco_a, reco_b images to files
        # the batched images should be arranged in rows of the resulting bigger image in this order: real_a, masked_a, reco_a into the file self.current_epoch_a.png
        # the batched images should be arranged in rows of the resulting bigger image in this order: real_b, masked_b, reco_b into the file self.current_epoch_b.png

        batch_size = preds.real_a.size(0)

        # For a: arrange real_a, masked_a, reco_a in rows
        grid_real_a = make_grid(preds.real_a, nrow=batch_size)
        grid_masked_a = make_grid(preds.masked_a, nrow=batch_size)
        grid_reco_a = make_grid(preds.reco_a, nrow=batch_size)
        big_image_a = torch.cat([grid_real_a, grid_masked_a, grid_reco_a], dim=1)
        save_image(big_image_a, os.path.join(subdir, f"{self.current_epoch}_a.png"))

        # For b: arrange real_b, masked_b, reco_b in rows
        grid_real_b = make_grid(preds.real_b, nrow=batch_size)
        grid_masked_b = make_grid(preds.masked_b, nrow=batch_size)
        grid_reco_b = make_grid(preds.reco_b, nrow=batch_size)
        big_image_b = torch.cat([grid_real_b, grid_masked_b, grid_reco_b], dim=1)
        save_image(big_image_b, os.path.join(subdir, f"{self.current_epoch}_b.png"))

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='train_')
        if self.current_epoch % 100 == 0 and batch_idx == 0 and self.hparams.args_dict.get("outdir"):
            self.log_preds(preds, self.hparams.args_dict.outdir)

        return losses['final']

    def validation_step(self, batch, batch_idx):
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='val_')

    def test_step(self, batch, batch_idx):
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='test_')