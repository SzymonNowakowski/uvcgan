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

######## SYNTHETIC PREPRARATION WITH CURRICULUM LEARNING ########

        preds.pure_synthetic = batch[0]
        preds.noised_synthetic = preds.pure_synthetic * batch[2]

        """
        TRAINING SCHEDULE: Stochastic Noise Interpolation
        Total Duration: self.hparams.args_dict.epochs (e.g., 4000)
        
        | Phase             | Epochs      | Interpolation Strategy (Alpha sampling)      | Objective                                 |
        |-------------------|-------------|---------------------------------------------|-------------------------------------------|
        | I: Ramp-Up        | 0 – 1500    | alpha ~ Uniform(0, max_alpha)               | Introduce noise of varying intensities.   |
        |                   |             | max_alpha: 0.0 -> 1.0                       |                                           |
        | II: Ramp-Down     | 1500 – 3000 | alpha ~ Uniform(min_alpha, 1.0)             | Phase out clean/low-noise samples.        |
        |                   |             | min_alpha: 0.0 -> 1.0                       |                                           |
        | III: Pure Noisy   | 3000 – 4000 | alpha = 1.0 (Fixed)                         | Converge on the target noisy distribution.|
        """

        epoch = self.current_epoch
        batch_size = preds.pure_synthetic.shape[0]

        # --- 1. Calculate Phase-Specific Alpha Bounds ---
        if epoch < 0:
            # Phase I: max_alpha grows 0 -> 1. Sampling Range: [0, max_alpha]
            max_alpha = epoch / 1500
            alpha_samples = torch.rand(batch_size, device=self.device) * max_alpha

        elif epoch < 0:
            # Phase II: min_alpha grows 0 -> 1. Sampling Range: [min_alpha, 1.0]
            progress = (epoch - 1500) / 1500
            min_alpha = progress
            alpha_samples = min_alpha + (torch.rand(batch_size, device=self.device) * (1.0 - min_alpha))

        else:
            # Phase III: Full noise for everyone
            alpha_samples = torch.ones(batch_size, device=self.device)

        # --- 2. Reshape alpha for broadcasting ---
        # Reshapes from [batch] to [batch, 1, 1, 1] to match image dimensions
        alpha_mask = alpha_samples.view(batch_size, *([1] * (len(preds.pure_synthetic.shape) - 1)))

        # --- 3. Execute Stochastic Interpolation ---
        # Every image in the batch now has a different noise intensity level
        preds.real_synthetic = (1 - alpha_mask) * preds.pure_synthetic + alpha_mask * preds.noised_synthetic
############### REAL PREPARATION ###########################

        preds.real_experimental = batch[1]

        preds.masked_synthetic = self.masking(preds.real_synthetic)
        preds.masked_experimental = self.masking(preds.real_experimental)

        preds.reconstruction_synthetic = self.generator_synthetic(preds.masked_synthetic)
        preds.reconstruction_experimental = self.generator_experimental(preds.masked_experimental)

        loss_synthetic = self.loss(preds.real_synthetic, preds.reconstruction_synthetic)
        loss_experimental = self.loss(preds.real_experimental, preds.reconstruction_experimental)


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

    def save_images(self, preds, subdir):
        self.save_image_group([preds.real_synthetic, preds.masked_synthetic, preds.reconstruction_synthetic, preds.pure_synthetic], os.path.join(subdir, f"{self.current_epoch}_synthetic.png"))
        self.save_image_group([preds.real_experimental, preds.masked_experimental, preds.reconstruction_experimental], os.path.join(subdir, f"{self.current_epoch}_experimental.png"))

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

