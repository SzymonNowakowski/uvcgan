import os

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid, save_image

from bio_PLB.models.abstract_model import AbstractModel
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate
from lightning_fabric.utilities.data import AttributeDict


from bio_PLB.tools import get_git_revision_short_hash


class AutoencoderOneWayWrapper(AbstractModel):
    def __init__(self, args_dict):
        super().__init__(args_dict)

        self.generator_synthetic = instantiate(args_dict.generator.model)
        init_weights(self.generator_synthetic, args_dict.generator.weight_init)

        self.loss = instantiate(args_dict.loss)

        self.masking = instantiate(args_dict.masking)


    def process_batch_supervised(self, batch):
        """get predictions, losses and mean errors (MAE)"""
        # execute the logic from uvcgan/cgan/autoencoder.py set_input & froward functions

        preds = AttributeDict()

        preds.real_synthetic = batch

        preds.masked_synthetic = self.masking(preds.real_synthetic)

        preds.reconstruction_synthetic = self.generator_synthetic(preds.masked_synthetic)

        loss_synthetic = self.loss(preds.real_synthetic, preds.reconstruction_synthetic)


        losses  = { 'loss_synthetic': loss_synthetic,
                    'final': loss_synthetic
                  }

        metrics = { }

        return preds, losses, metrics

    def save_images(self, preds, subdir):
        self.save_image_group([preds.real_synthetic, preds.masked_synthetic, preds.reconstruction_synthetic], os.path.join(subdir, f"{self.current_epoch}_synthetic.png"))

