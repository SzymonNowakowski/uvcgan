import os

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid, save_image

from bio_PLB.models.autoencoder_one_way_wrapper import AutoencoderOneWayWrapper
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate
from lightning_fabric.utilities.data import AttributeDict


from bio_PLB.tools import get_git_revision_short_hash


class AutoencoderTwoWayWrapper(AutoencoderOneWayWrapper):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        # it has an additional head:
        self.generator_experimental = instantiate(args_dict.generator.model)
        init_weights(self.generator_experimental, args_dict.generator.weight_init)


    def process_batch_supervised(self, batch):
        """get predictions, losses and mean errors (MAE)"""
        # execute the logic from uvcgan/cgan/autoencoder.py set_input & froward functions

        preds = AttributeDict()

        #image_types = ['real_a', 'reco_a', 'real_b', 'reco_b', 'masked_a', 'masked_b']
        #for image_type in image_types:
        #    preds[image_type] = None

        preds.pure_synthetic = batch[0].to(self.device)

        # overlaying random background
        # over the synthetic image
        preds.real_synthetic = preds.pure_synthetic * batch[2].to(self.device)
        preds.real_experimental = batch[1].to(self.device)

        preds.masked_synthetic = self.masking(preds.real_synthetic)
        preds.masked_experimental = self.masking(preds.real_experimental)

        preds.reco_synthetic = self.generator_synthetic(preds.masked_synthetic)
        preds.reco_experimental = self.generator_experimental(preds.masked_experimental)

        loss_synthetic = self.loss(preds.real_synthetic, preds.reco_synthetic)
        loss_experimental = self.loss(preds.real_experimental, preds.reco_experimental)


        losses  = { 'loss_synthetic': loss_synthetic,
                    'loss_experimental': loss_experimental,
                    'final': loss_synthetic + loss_experimental
                  }

        metrics = { }

        return preds, losses, metrics

    def log_preds(self, preds, outdir):
        # make a new subdirectory in outdir - if not already - with get_git_revision_short_hash()
        subdir = os.path.join(outdir, get_git_revision_short_hash())
        os.makedirs(subdir, exist_ok=True)
        # for preds dictionary, save the batch of real_a, real_b, masked_a, masked_b, reco_a, reco_b images to files
        # the file self.current_epoch_xxx.png:
        #   the batched images should be arranged in a column and rows of the resulting bigger image should be in this order: real_xxx, masked_xxx, reco_xxx

        self.save_image_group(preds.real_synthetic, preds.masked_synthetic, preds.reco_synthetic, os.path.join(subdir, f"{self.current_epoch}_synthetic.png"), preds.pure_synthetic)
        self.save_image_group(preds.real_experimental, preds.masked_experimental, preds.reco_experimental, os.path.join(subdir, f"{self.current_epoch}_experimental.png"))

    def transplant_experimental_head(self):
        """
        Copies the weights from the trained synthetic generator to the
        experimental generator. This ensures both heads start from the
        same pre-trained state before fine-tuning on experimental data.
        """
        # Access the state dict of the synthetic generator
        synthetic_state = self.generator_synthetic.state_dict()

        # Load the state dict into the experimental generator
        # We use strict=True here because the architectures should be identical
        self.generator_experimental.load_state_dict(synthetic_state)

