import os

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid, save_image

from bio_PLB.models.abstract_model import AbstractModel
from bio_PLB.models.autoencoder_two_way_wrapper import AutoencoderTwoWayWrapper
from bio_PLB.models.cyclegan_wrapper import CycleGANWrapper
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate
from lightning_fabric.utilities.data import AttributeDict


from bio_PLB.tools import get_git_revision_short_hash


class CycleGANPrediscriminatorWrapper(CycleGANWrapper):
    def __init__(self, args_dict):
        super().__init__(args_dict)

        # networks:
            # prediscriminators
        self.prediscriminator_synthetic = instantiate(args_dict.generator.model)
        init_weights(self.prediscriminator_synthetic, args_dict.generator.weight_init)

        self.prediscriminator_experimental = instantiate(args_dict.generator.model)
        init_weights(self.prediscriminator_experimental, args_dict.generator.weight_init)

        # linking:
        self.discriminator_synthetic.prediscriminator = self.prediscriminator_synthetic
        self.discriminator_experimental.prediscriminator = self.prediscriminator_experimental


    def compute_discriminator_loss(self, discriminator_model, image, torch_init_like_fun):
        def concatenate_channelwise(imageA, imageB):
            # image: (B, C, H, W), features: (B, C, H, W) -> result: (B, 2*C, H, W)
            return torch.cat((imageA, imageB), dim=-3)


        discriminator_prediction = discriminator_model(concatenate_channelwise(image, discriminator_model.prediscriminator(image)))


        loss = self.discriminator_loss(discriminator_prediction, torch_init_like_fun(discriminator_prediction))
        return loss

    def process_batch_supervised(self, batch):
        if self.current_epoch == 10:
            self.lambda_discriminator = 0   #the discriminator will get strong enough

        super().process_batch_supervised()

    def transplant_prediscriminator_heads(self, donor_synthetic: AutoencoderTwoWayWrapper, donor_experimental: AutoencoderTwoWayWrapper):

        state = donor_synthetic.generator_synthetic.state_dict()
        self.prediscriminator_synthetic.load_state_dict(state)

        state = donor_experimental.generator_experimental.state_dict()
        self.prediscriminator_experimental.load_state_dict(state)

    def reinstaintiate_discriminator(self):
        self.discriminator_synthetic = instantiate(self.hparams.args_dict.discriminator.model)
        init_weights(self.discriminator_synthetic, self.hparams.args_dict.discriminator.weight_init)

        self.discriminator_experimental = instantiate(self.hparams.args_dict.discriminator.model)
        init_weights(self.discriminator_experimental, self.hparams.args_dict.discriminator.weight_init)

        # relinking:
        self.discriminator_synthetic.prediscriminator = self.prediscriminator_synthetic
        self.discriminator_experimental.prediscriminator = self.prediscriminator_experimental