import os

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid, save_image

from bio_PLB.models.abstract_model import AbstractModel
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate
from lightning_fabric.utilities.data import AttributeDict


from bio_PLB.tools import get_git_revision_short_hash


class CycleGANWrapper(AbstractModel):
    def __init__(self, args_dict):
        super().__init__(args_dict)

        #networks
        self.generator_synthetic2experimental = instantiate(args_dict.generator.model)
        init_weights(self.generator_synthetic2experimental, args_dict.generator.weight_init)

        self.generator_experimental2synthetic = instantiate(args_dict.generator.model)
        init_weights(self.generator_experimental2synthetic, args_dict.generator.weight_init)

        self.discriminator_synthetic = instantiate(args_dict.discriminator.model)
        init_weights(self.discriminator_synthetic, args_dict.discriminator.weight_init)

        self.discriminator_experimental = instantiate(args_dict.discriminator.model)
        init_weights(self.discriminator_experimental, args_dict.discriminator.weight_init)

        #losses
        self.identity_loss = instantiate(args_dict.identity_loss)
        self.discriminator_loss  = instantiate(args_dict.discriminator_loss)

        #lambdas
        self.lambda_preserve_identity = args_dict.lambda_preserve_identity
        self.lambda_cycle_identity = args_dict.lambda_cycle_identity
        self.lambda_generator = args_dict.lambda_generator
        self.lambda_discriminator = args_dict.lambda_discriminator


    def discriminator_loss(self, discriminator_model, image, torch_init_like_fun):
        discriminator_prediction = discriminator_model(image)
        loss = self.discriminator_loss(discriminator_prediction, torch_init_like_fun(discriminator_prediction))
        return loss

    def set_requires_grad(self, discriminator_model, requires_grad):
        for param in discriminator_model.parameters():
            param.requires_grad = requires_grad

    def process_batch_supervised(self, batch):

        preds = AttributeDict()

        self.set_requires_grad(self.discriminator_experimental, False)
        self.set_requires_grad(self.discriminator_synthetic, False)

        preds.pure_synthetic = batch[0]
        preds.real_synthetic = preds.pure_synthetic * batch[2]
        preds.real_experimental = batch[1]

        preds.fake_experimental = self.generator_synthetic2experimental(preds.real_synthetic)
        preds.reconstruction_synthetic = self.generator_experimental2synthetic(preds.fake_experimental)

        preds.fake_synthetic = self.generator_experimental2synthetic(preds.real_experimental)
        preds.reconstruction_experimental = self.generator_synthetic2experimental(preds.fake_synthetic)

        #preserve_identity measures the amount of changes a generator adds to an image which already is in a correct domain
        loss_preserve_identity_synthetic = self.identity_loss(preds.real_synthetic, self.generator_experimental2synthetic(preds.real_synthetic))
        loss_preserve_identity_experimental = self.identity_loss(preds.real_experimental, self.generator_synthetic2experimental(preds.real_experimental))

        # cycle measures the amount of changes that are added after going a one full cycle, either way
        loss_cycle_identity_synthetic = self.identity_loss(preds.real_synthetic, preds.reconstruction_synthetic)
        loss_cycle_identity_experimental = self.identity_loss(preds.real_experimental, preds.reconstruction_experimental)

        loss_generator_synthetic = self.discriminator_loss(self.discriminator_synthetic, preds.fake_synthetic, torch.ones_like)
        loss_generator_experimental = self.discriminator_loss(self.discriminator_experimental, preds.fake_experimental, torch.ones_like)

        self.set_requires_grad(self.discriminator_experimental, True)
        self.set_requires_grad(self.discriminator_synthetic, True)

        loss_discriminator_synthetic_fake = self.discriminator_loss(self.discriminator_synthetic, preds.fake_synthetic.detach(), torch.zeros_like)
        loss_discriminator_synthetic_real = self.discriminator_loss(self.discriminator_synthetic, preds.real_synthetic, torch.ones_like)
        loss_discriminator_experimental_fake = self.discriminator_loss(self.discriminator_experimental, preds.fake_experimental.detach(), torch.zeros_like)
        loss_discriminator_experimental_real = self.discriminator_loss(self.discriminator_experimental, preds.real_experimental, torch.ones_like)


        losses  = { 'preserve_identity_synthetic': loss_preserve_identity_synthetic,
                    'preserve_identity_experimental': loss_preserve_identity_experimental,
                    'cycle_identity_synthetic': loss_cycle_identity_synthetic,
                    'cycle_identity_experimental': loss_cycle_identity_experimental,
                    'generator_synthetic': loss_generator_synthetic,
                    'generator_experimental': loss_generator_experimental,
                    'discriminator_synthetic_fake': loss_discriminator_synthetic_fake,
                    'discriminator_synthetic_real': loss_discriminator_synthetic_real,
                    'discriminator_experimental_fake': loss_discriminator_experimental_fake,
                    'discriminator_experimental_real': loss_discriminator_experimental_real,
                    'final': self.lambda_preserve_identity * loss_preserve_identity_synthetic +
                             self.lambda_preserve_identity * loss_preserve_identity_experimental +
                             self.lambda_cycle_identity * loss_cycle_identity_synthetic +
                             self.lambda_cycle_identity * loss_cycle_identity_experimental +
                             self.lambda_generator * loss_generator_synthetic +
                             self.lambda_generator * loss_generator_experimental +
                             self.lambda_discriminator * loss_discriminator_synthetic_fake +
                             self.lambda_discriminator * loss_discriminator_synthetic_real +
                             self.lambda_discriminator * loss_discriminator_experimental_fake +
                             self.lambda_discriminator * loss_discriminator_experimental_real
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

        self.save_image_group([preds.real_synthetic, preds.fake_experimental, preds.reconstruction_synthetic], os.path.join(subdir, f"{self.current_epoch}_synthetic.png"))
        self.save_image_group([preds.real_experimental, preds.fake_synthetic, preds.reconstruction_experimental], os.path.join(subdir, f"{self.current_epoch}_experimental.png"))


    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='train_')
        if (self.current_epoch == 0 or self.current_epoch % 1000 == 999) and batch_idx == 0 and self.hparams.args_dict.get("outdir"):
            self.log_preds(preds, self.hparams.args_dict.outdir)

        return losses['final']