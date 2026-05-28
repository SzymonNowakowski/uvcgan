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

        self.probability_flip_labels_discriminator = args_dict.get("probability_flip_labels_discriminator", 0.5)

        self.automatic_optimization = False


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

    def compute_discriminator_loss(self, discriminator_model, image, compute_labels_fun):
        discriminator_prediction = discriminator_model(image)
        loss = self.discriminator_loss(discriminator_prediction, compute_labels_fun(discriminator_prediction))
        return loss

    def set_requires_grad(self, discriminator_model, requires_grad):
        for param in discriminator_model.parameters():
            param.requires_grad = requires_grad

    def close_to_zeros_with_flip(self, tensor):
        # produce a tensor shaped like input tensor
        # with values close to 0 (0-0.3) randomly drawn from uniform distribution
        # and with a certain probability (self.probability_flip_labels_discriminator) flipped to value
        # between 0.7 and 1.0 (close to 1)
        mask = torch.rand(tensor.size())
        return ((mask < self.probability_flip_labels_discriminator) * (0.7 + torch.rand(tensor.size()) * 0.3) + (mask >= self.probability_flip_labels_discriminator) * torch.rand(tensor.size()) * 0.3).to(tensor.device)

    def close_to_ones_with_flip(self, tensor):
        # produce a tensor shaped like input tensor
        # with values close to 1 (0.7-1.0) randomly drawn from uniform distribution
        # and with a certain probability (self.probability_flip_labels_discriminator) flipped to value
        # between 0.0 and 0.3 (close to 0)
        mask = torch.rand(tensor.size())
        return ((mask >= self.probability_flip_labels_discriminator) * (0.7 + torch.rand(tensor.size()) * 0.3) + (mask < self.probability_flip_labels_discriminator) * torch.rand(tensor.size()) * 0.3).to(tensor.device)

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

        loss_generator_synthetic = self.compute_discriminator_loss(self.discriminator_synthetic, preds.fake_synthetic, torch.ones_like)
        loss_generator_experimental = self.compute_discriminator_loss(self.discriminator_experimental, preds.fake_experimental, torch.ones_like)

        self.set_requires_grad(self.discriminator_experimental, True)
        self.set_requires_grad(self.discriminator_synthetic, True)

        loss_discriminator_synthetic_fake = self.compute_discriminator_loss(self.discriminator_synthetic, preds.fake_synthetic.detach(), self.close_to_zeros_with_flip)
        loss_discriminator_synthetic_real = self.compute_discriminator_loss(self.discriminator_synthetic, preds.real_synthetic, self.close_to_ones_with_flip)
        loss_discriminator_experimental_fake = self.compute_discriminator_loss(self.discriminator_experimental, preds.fake_experimental.detach(), self.close_to_zeros_with_flip)
        loss_discriminator_experimental_real = self.compute_discriminator_loss(self.discriminator_experimental, preds.real_experimental, self.close_to_ones_with_flip)


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

    def save_images(self, preds, subdir):
        self.save_image_group([preds.real_synthetic, preds.fake_experimental, preds.reconstruction_synthetic], os.path.join(subdir, f"{self.current_epoch}_synthetic.png"))
        self.save_image_group([preds.real_experimental, preds.fake_synthetic, preds.reconstruction_experimental], os.path.join(subdir, f"{self.current_epoch}_experimental.png"))


    def transplant_generator_heads(self, donor_synthetic: AutoencoderTwoWayWrapper, donor_experimental: AutoencoderTwoWayWrapper):

        state = donor_synthetic.generator_synthetic.state_dict()
        self.generator_synthetic2experimental.load_state_dict(state)

        state = donor_experimental.generator_experimental.state_dict()
        self.generator_experimental2synthetic.load_state_dict(state)