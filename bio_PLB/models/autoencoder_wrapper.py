import pytorch_lightning as pl
import torch
from uvcgan.base.weight_init import init_weights

from hydra.utils import instantiate


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
        self.save_hyperparameters()

    def configure_optimizers(self):
        # 1. Instantiate the optimizer, passing model parameters
        optimizer = instantiate(
            self.hparams.generator.optimizer,
            params=self.parameters()
        )

        # 2. Instantiate the scheduler, passing the optimizer
        scheduler = instantiate(
            self.hparams.scheduler,
            optimizer=optimizer
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or "epoch"
            },
        }

    def forward(self, x):
        return #TODO

    def process_batch_supervised(self, batch):
        """get predictions, losses and mean errors (MAE)"""
        inputs, targets = batch
        inputs  = inputs.to(self.device)
        targets = targets.to(self.device)

        preds = self(inputs)

        loss    = self.loss(preds, targets)

        #metric_fn = torch.nn.L1Loss()
        #metric    = metric_fn(preds, targets)

        losses  = { 'final': loss }
        metrics = { }#'mae': metric }

        return preds, losses, metrics


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