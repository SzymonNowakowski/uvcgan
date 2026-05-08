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

######## SYNTHETIC PREPRARATION WITH CURRICULUM LEARNING ########

        preds.pure_synthetic = batch[0].to(self.device)
        preds.noised_synthetic = preds.pure_synthetic * batch[2].to(self.device)

        """
        TRAINING SCHEDULE: From Synthetic to Noisy Data (Curriculum Learning)
        Total Duration: self.hparams.args_dict.epochs
        Current epoch: self.current_epoch

        | Phase            | Epochs      | Alpha (Interpolation) | P_noise (Mix)  | Objective                                |
        |------------------|-------------|-----------------------|----------------|------------------------------------------|
        | I: Adaptation    | 0 – 1000    | 0.0 -> 0.5 (Linear)   | 0% -> 50%      | Gentle introduction of noise/error.      |
        | II: Transition   | 1000 – 2500 | 0.5 -> 1.0 (Linear)   | 50% -> 80%     | Core training on noisy distribution.     |
        | III: Consolidation| 2500 – 3500 | 1.0 (Constant)        | 80% -> 100%    | Phasing out reliance on clean data.      |
        | IV: Fine-tuning  | 3500 – 4000 | 1.0 (Constant)        | 100%           | Stabilization on target noisy distribution.|

        Parameters:
        - Alpha: Interpolation weight between clean and noisy samples.
        - P_noise: Percentage of samples in a batch that are noisy.
        """

        # current training progress
        epoch = self.current_epoch

        # phase 1: adaptation (0-1000)
        if epoch < 1000:
            alpha = (epoch / 1000) * 0.5
            p_noise = (epoch / 1000) * 0.5
        # phase 2: transition (1000-2500)
        elif epoch < 2500:
            # normalize progress between 0 and 1 for this phase
            progress = (epoch - 1000) / 1500
            alpha = 0.5 + (progress * 0.5)
            p_noise = 0.5 + (progress * 0.3)
        # phase 3: consolidation (2500-3500)
        elif epoch < 3500:
            progress = (epoch - 2500) / 1000
            alpha = 1.0
            p_noise = 0.8 + (progress * 0.2)
        # phase 4: fine-tuning (3500+)
        else:
            alpha = 1.0
            p_noise = 1.0

        # create interpolation for all samples in the batch
        # alpha controls the "intensity" of the noise
        interpolated_samples = (1 - alpha) * preds.pure_synthetic + alpha * preds.noised_synthetic

        # create a mask based on p_noise to decide which samples stay pure and which are noised
        # batch[0].shape[0] is the batch size
        batch_size = preds.pure_synthetic.shape[0]
        noise_mask = (torch.rand(batch_size, device=self.device) < p_noise).float()

        # reshape mask for element-wise multiplication (e.g., [batch, 1, 1, 1] for images)
        for _ in range(len(preds.pure_synthetic.shape) - 1):
            noise_mask = noise_mask.unsqueeze(-1)

        # mix pure and interpolated samples based on the p_noise probability
        preds.real_synthetic = (noise_mask * interpolated_samples) + ((1 - noise_mask) * preds.pure_synthetic)

############### REAL PREPARATION ###########################

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

