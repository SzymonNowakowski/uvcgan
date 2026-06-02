import os

import pytorch_lightning as pl
from omegaconf import ListConfig

import torch
from torchvision.utils import make_grid, save_image

from bio_PLB.models.abstract_model import AbstractModel
from bio_PLB.models.autoencoder_two_way_wrapper import AutoencoderTwoWayWrapper
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate
from lightning_fabric.utilities.data import AttributeDict

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

        self.automatic_optimization = False

    #instead of self.lambdas_blablabla define a set of properties that will serve as polymorphic methods and enable dynamic linking in case of loading a network from a checkpoint
    @property
    def lambda_preserve_identity(self):
        #linear growth from 0.0 to max over self.hparams.args_dict.lambda_growth_epochs
        current_level = min(self.current_epoch, self.hparams.args_dict.lambda_growth_epochs) / self.hparams.args_dict.lambda_growth_epochs * self.hparams.args_dict.lambda_preserve_identity
        return current_level

    @property
    def lambda_cycle_identity(self):
        # linear growth from 0.0 to max over self.hparams.args_dict.lambda_growth_epochs
        current_level = min(self.current_epoch, self.hparams.args_dict.lambda_growth_epochs) / self.hparams.args_dict.lambda_growth_epochs * self.hparams.args_dict.lambda_cycle_identity
        return current_level

    @property
    def lambda_generator(self):
        return self.hparams.args_dict.lambda_generator

    @property
    def lambda_discriminator(self):
        return self.hparams.args_dict.lambda_discriminator

    @property
    def lambda_gradient_penalty(self):
        return self.hparams.args_dict.lambda_gradient_penalty

    @property
    def probability_flip_labels_discriminator(self):
        return self.hparams.args_dict.probability_flip_labels_discriminator

    def generator_params(self):
        return list(self.generator_synthetic2experimental.parameters()) + \
                     list(self.generator_experimental2synthetic.parameters())

    def discriminator_params(self):
        return list(self.discriminator_synthetic.parameters()) + \
                      list(self.discriminator_experimental.parameters())

    def configure_optimizers(self):
        from hydra.utils import instantiate

        gen_params = self.generator_params()

        disc_params = self.discriminator_params()

        param_groups = [gen_params, disc_params]

        opt_configs = self.hparams.args_dict.optimizer
        optimizers = []
        for i, params in enumerate(param_groups):
            # i-th config or (if not a list) the same config for all groups
            if isinstance(opt_configs, ListConfig):
                config = opt_configs[i]
            elif isinstance(opt_configs, list):
                config = opt_configs[i]
            else:
                config = opt_configs

            opt = instantiate(config, params=params)
            optimizers.append(opt)

        # the same for schedulers
        if self.hparams.args_dict.get("scheduler"):
            sched_configs = self.hparams.args_dict.scheduler
            schedulers = []
            for i, opt in enumerate(optimizers):
                if isinstance(sched_configs, ListConfig):
                    config = sched_configs[i]
                elif isinstance(sched_configs, list):
                    config = sched_configs[i]
                else:
                    config = sched_configs

                sch = instantiate(config, optimizer=opt)
                schedulers.append({
                    "scheduler": sch,
                    "interval": "epoch"
                })
            return optimizers, schedulers

        return optimizers

    # manual handling of optimization
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        loss_final = super().training_step(batch, batch_idx)  # this will call process_batch_supervised and log the losses and metrics

        opt_g, opt_d = self.optimizers()

        opt_g.zero_grad()
        opt_d.zero_grad()

        self.manual_backward(loss_final)

        #self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        #self.clip_gradients(opt_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        if self.current_epoch % 10 == 9:
            opt_g.step()

        opt_d.step()

        # manual handling of "step" based schedulers
        for config in self.trainer.lr_scheduler_configs:
            if config.interval == 'step':
                config.scheduler.step()

        return loss_final

    # manual handling of "epoch" based schedulers
    def on_train_epoch_end(self):
        for config in self.trainer.lr_scheduler_configs:
            if config.interval == 'epoch':
                config.scheduler.step()

    def compute_discriminator_prediction(self, discriminator_model, image):
        return discriminator_model(image)

    def compute_discriminator_loss(self, discriminator_model, image, objective):
        discriminator_prediction = self.compute_discriminator_prediction(discriminator_model, image)

        if self.hparams.args_dict.get("gan_type") == "wasserstein":
            wgan_sign_map = {
                'real': -1.0,
                'fake': 1.0,
                'generator': -1.0
            }
            loss = wgan_sign_map[objective] * discriminator_prediction.mean()
        else:
            gan_label_functions = {
                'real': self.close_to_ones_with_flip,
                'fake': self.close_to_zeros_with_flip,
                'generator': torch.ones_like,
            }
            loss = self.discriminator_loss(discriminator_prediction, gan_label_functions[objective](discriminator_prediction))
        return loss

    def compute_gradient_penalty(self,
            discriminator_model, real_data, fake_data, constant=1.0, epsilon = 1e-16
    ):
        """Rewritten and improved from uvcgan/base/losses.py cal_gradient_penalty function


        source: https://arxiv.org/abs/1704.00028

        Arguments:
            discriminator_model (network)              -- discriminator network
            real_data (tensor array)    -- real images
            fake_data (tensor array)    -- generated images from the generator
            device (str)                -- torch device
            constant (float)            -- the constant used in formula:
                (||gradient||_2 - constant)^2 / constant^2

        Returns the gradient penalty loss
        """
        alpha = torch.rand(real_data.shape[0], 1, device=real_data.device)
        alpha = alpha.expand(
            real_data.shape[0], real_data.nelement() // real_data.shape[0]
        ).contiguous().view(*real_data.shape)

        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)

        interpolatesv.requires_grad_(True)

        disc_interpolates = self.compute_discriminator_prediction(discriminator_model, interpolatesv)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolatesv,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, retain_graph=True, only_inputs=True
        )

        gradients = gradients[0].view(real_data.size(0), -1)

        gradient_penalty = (
                (((gradients + epsilon).norm(2, dim=1) - constant) ** 2)/(constant**2)   #division by (constant**2) after the Karras' https://arxiv.org/abs/1710.10196 paper
                           ).mean()

        return gradient_penalty

    def set_requires_grad(self, discriminator_model, requires_grad):
        for param in discriminator_model.parameters():
            param.requires_grad = requires_grad

    def close_to_zeros_with_flip(self, tensor):
        # produce a tensor shaped like input tensor
        # with values close to 0 (0-0.3) randomly drawn from uniform distribution
        # and with a certain probability (self.probability_flip_labels_discriminator) flipped to value
        # between 0.7 and 1.0 (close to 1)
        mask = torch.rand(tensor.size(), device=tensor.device)
        return ((mask < self.probability_flip_labels_discriminator) * (0.7 + torch.rand(tensor.size(), device=tensor.device) * 0.3) + (mask >= self.probability_flip_labels_discriminator) * torch.rand(tensor.size(), device=tensor.device) * 0.3)

    def close_to_ones_with_flip(self, tensor):
        # produce a tensor shaped like input tensor
        # with values close to 1 (0.7-1.0) randomly drawn from uniform distribution
        # and with a certain probability (self.probability_flip_labels_discriminator) flipped to value
        # between 0.0 and 0.3 (close to 0)
        mask = torch.rand(tensor.size(), device=tensor.device)
        return ((mask >= self.probability_flip_labels_discriminator) * (0.7 + torch.rand(tensor.size(), device=tensor.device) * 0.3) + (mask < self.probability_flip_labels_discriminator) * torch.rand(tensor.size(), device=tensor.device) * 0.3)

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

        loss_generator_synthetic = self.compute_discriminator_loss(self.discriminator_synthetic, preds.fake_synthetic, 'generator')
        loss_generator_experimental = self.compute_discriminator_loss(self.discriminator_experimental, preds.fake_experimental, 'generator')

        self.set_requires_grad(self.discriminator_experimental, True)
        self.set_requires_grad(self.discriminator_synthetic, True)

        loss_discriminator_synthetic_fake = self.compute_discriminator_loss(self.discriminator_synthetic, preds.fake_synthetic.detach(), 'fake')
        loss_discriminator_synthetic_real = self.compute_discriminator_loss(self.discriminator_synthetic, preds.real_synthetic, 'real')
        loss_discriminator_experimental_fake = self.compute_discriminator_loss(self.discriminator_experimental, preds.fake_experimental.detach(), 'fake')
        loss_discriminator_experimental_real = self.compute_discriminator_loss(self.discriminator_experimental, preds.real_experimental, 'real')

        loss_gradient_penalty_synthetic = self.compute_gradient_penalty(self.discriminator_synthetic, preds.real_synthetic, preds.fake_synthetic.detach())
        loss_gradient_penalty_experimental = self.compute_gradient_penalty(self.discriminator_experimental, preds.real_experimental, preds.fake_experimental.detach())

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
                    'discriminator_gradient_penalty_synthetic': loss_gradient_penalty_synthetic,
                    'discriminator_gradient_penalty_experimental': loss_gradient_penalty_experimental,

                    'final': self.lambda_preserve_identity * loss_preserve_identity_synthetic +
                             self.lambda_preserve_identity * loss_preserve_identity_experimental +
                             self.lambda_cycle_identity * loss_cycle_identity_synthetic +
                             self.lambda_cycle_identity * loss_cycle_identity_experimental +
                             self.lambda_generator * loss_generator_synthetic +
                             self.lambda_generator * loss_generator_experimental +
                             self.lambda_discriminator * loss_discriminator_synthetic_fake +
                             self.lambda_discriminator * loss_discriminator_synthetic_real +
                             self.lambda_discriminator * loss_discriminator_experimental_fake +
                             self.lambda_discriminator * loss_discriminator_experimental_real +
                             self.lambda_gradient_penalty * loss_gradient_penalty_synthetic +
                             self.lambda_gradient_penalty * loss_gradient_penalty_experimental

                  }

        metrics = { }

        return preds, losses, metrics

    def save_images(self, preds, subdir):
        self.save_image_group([preds.real_synthetic, preds.fake_experimental, preds.reconstruction_synthetic], os.path.join(subdir, f"{self.current_epoch}_synthetic.png"))
        self.save_image_group([preds.real_experimental, preds.fake_synthetic, preds.reconstruction_experimental], os.path.join(subdir, f"{self.current_epoch}_experimental.png"))


    def transplant_generator_heads(self, donor_synthetic: AutoencoderTwoWayWrapper, donor_experimental: AutoencoderTwoWayWrapper):

        state = donor_synthetic.generator_synthetic.state_dict()
        self.generator_synthetic2experimental.load_state_dict(state)

        state = donor_experimental.generator_experimental.state_dict()
        self.generator_experimental2synthetic.load_state_dict(state)