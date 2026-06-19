import os

from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from bio_PLB.models.autoencoder_two_way_wrapper import AutoencoderTwoWayWrapper
from bio_PLB.models.cyclegan_prediscriminator_wrapper import CycleGANPrediscriminatorWrapper
#from torchvision.transforms import ToTensor

from bio_PLB.models.cyclegan_wrapper import CycleGANWrapper

#from bio_PLB.external.PLB.regression.src.plbregression.coordinator_dataset import SyntheticDatasetAdapter

import pytorch_lightning as pl
import torch

import bio_PLB.tools
from pytorch_lightning import seed_everything

def main():
    # Seeds random, numpy, torch, torch.cuda, and dataloader workers for reproducibility
    seed_everything(42, workers=True)

    print("Current Working Directory:", os.getcwd())

    # register a resolver named "eval"
    # this tells OmegaConf: "When you see ${eval: ...}, run it through Python's eval"
    OmegaConf.register_new_resolver("eval", eval)

    args_dict = OmegaConf.create({
        'epochs': 2000,
        'gan_type': 'wasserstein',    #values: 'wasserstein', 'gan'
        'update_generator_every': 2,
        'outdir': 'outdir',
        'save_images_every': 10,
        'batch_size': 32,
        'target_px': 160,
        'num_workers': 19,   #19 is the number of cores on the machine
        'data': {
            'dataset': {
                '_target_': 'bio_PLB.external.PLB.regression.src.plbregression.coordinator_dataset.CoordinatorDataset',
                # these are the values of normalization used internally in the Coordinator as coded before the refactor
                # TODO: now they need to be executed as externall transforms
                # image_real = GlobalAndInstanceNorm(global_mean=0.2363, global_std=0.1224)(image_real)
                # image_synth = GlobalAndInstanceNorm(global_mean=0.7367, global_std=0.1922)(image_synth)
                'datasets': [
                    {
                        '_target_': 'bio_PLB.external.PLB.regression.src.plbregression.coordinator_dataset.SyntheticDatasetAdapter',
                        'synthetic_dataset_instance': {
                            # Recursive instantiation of the external research dataset
                            '_target_': 'bio_PLB.external.PLB.regression.src.plbregression.synthetic_dataset.SyntheticDataset',
                            'data_dir': "data/synthetic2real/synthetic_0.5_px_nm/dataset_01_20260223/",
                            'transforms': [
                                {
                                    '_target_': 'bio_PLB.external.PLB.regression.src.plbregression.transforms.RandomRotatedShiftedCrop',
                                    'target_px': '${target_px}',
                                    'interpolation': 'cubic',
                                    'allow_background_crop': False,
                                    #TODO: add mean/std
                                },
                            # no microscopic noise
                            ],
                            'return_tensors': True,
                        }
                    },
                    {
                        '_target_': 'bio_PLB.external.PLB.regression.src.plbregression.experimental_dataset.ExperimentalDataset',
                        'image_dir': "data/synthetic2real/real/crop_2957",
                        'metadata_csv_path': "data/synthetic2real/real/data_summary_2957.csv",
                        'target_nm': "${eval:'2 * ${target_px}'}",
                        'target_px': '${target_px}',
                        'return_tensors': True,
                        'distribution': 'normal'   # fixed normal distribution in PLB code (commit #ca48670 in PLB Center4ML repository main branch)
                        # TODO: add mean/std
                    },
                    {
                        '_target_': 'bio_PLB.external.PLB.regression.src.plbregression.experimental_dataset.ExperimentalDataset',
                        'image_dir': "data/synthetic2real/backgrounds/tla_spireai",
                        'metadata_csv_path': "data/synthetic2real/backgrounds/background_files.csv",
                        'target_nm': "${eval:'2 * ${target_px}'}",
                        'target_px': '${target_px}',
                        'return_tensors': True,
                        'distribution': 'uniform' # the backgrounds should be sampled uniformly
                        # TODO: add mean/std
                    }
                ],
                'main_dataset': 1,  # experimental dataset is main
                #'shared_transforms': [
                #    {'_target_': 'torchvision.transforms.ToTensor'},
                #]
            },
        },
        'generator': {
            'synthetic_generator_link': 'logs/bert-two-way-160px-5c87673/checkpoints/best_synthetic_loss_epoch=3359-train_loss_synthetic_loss=0.01361-train_final_loss=0.02877.ckpt',
            'experimental_generator_link': 'logs/bert-two-way-160px-5c87673/checkpoints/best_experimental_loss_epoch=2100-train_loss_experimental_loss=0.01296-train_final_loss=0.02874.ckpt',
            'model': {
                # 'model' : 'vit-unet',
                '_target_': 'uvcgan.models.generator.vitunet.ViTUNetGenerator',
                'image_shape': (1, '${target_px}', '${target_px}'),
                'features': 96,#128,384,
                'n_heads': 4,#4,6
                'n_blocks': 4,#4,12
                'ffn_features': 384,#512,1536
                'embed_features': 96,#128,,384,
                'activ': 'gelu',
                'norm': 'layer',
                'unet_features_list': [12, 24, 48, 96],#[48, 96, 192, 384],
                'unet_activ': 'leakyrelu',
                'unet_norm': 'instance',
                'unet_downsample': 'conv',
                'unet_upsample': 'upsample-conv',
                'rezero': True,
                'activ_output': 'sigmoid',
            },
            'weight_init' : {
                    'name'      : 'normal',
                    'init_gain' : 0.02,
            }
        },
        'discriminator': {
            'discriminator_link': 'logs/cyclegan-ce19c7c/checkpoints/epoch_epoch=1024-train_final_loss=7.88303.ckpt',
            'model': {
                '_target_': 'uvcgan.base.networks.NLayerDiscriminator',
                'image_shape': (1, '${target_px}', '${target_px}'),
                # chanel is would be 2 for prediscriminator because of the concatenation of image and prediscriminator BERT features
                #'ndf': 16,                # the number of filters in the last conv layer
                #'n_layers': 2,            # the number of conv layers in the discriminator
                #'max_mult': 4,            # normalization layer
            },
            'weight_init': {
                'name': 'normal',
                'init_gain': 0.02,
            }
        },
    'optimizer': [
        {
        '_target_': 'torch.optim.AdamW',  # Generator optimization
        'lr': "${eval:'${batch_size} * 2e-3 / 512'}",
        'betas': (0.9, 0.99),
        'weight_decay': 0.05,
        },
        {
            '_target_': 'torch.optim.AdamW',  # Generator optimization
            'lr': "${eval:'${batch_size} * 2e-3 / 512'}",
            'betas': (0.9, 0.99),
            'weight_decay': 0.05,
        }
        #{
        #    '_target_': 'torch.optim.SGD',  # Dyscriminator optimization
        #    'lr': "${eval:'${batch_size} * 2e-3 / 512'}",
        #    'momentum': 0.9,
        #}

    ],
    #'warmup_epochs': 100,
    'scheduler': None, #{
        #'_target_': 'torch.optim.lr_scheduler.LambdaLR',
        #'lr_lambda': "${eval:'lambda epoch: min(1.0, (epoch+1) / ${warmup_epochs})'}"
    #},
    'identity_loss'     : {'_target_' : 'torch.nn.L1Loss'},
    'discriminator_loss': {'_target_': 'torch.nn.BCEWithLogitsLoss'},
    'lambda_preserve_identity': 0.1,
    'lambda_cycle_identity': 0.1,
    'lambda_growth_epochs': 100, # >0; number of epochs it takes for identity- and cycle- lambdas to reach the max levels
    'lambda_generator': 1.0,
    'lambda_discriminator': 1.0,
    'lambda_gradient_penalty': 10.0,
    'lambda_gradient_decay_epochs': 0,
    'lambda_gradient_decayed_epochs': 0,
    'probability_flip_labels_discriminator': 0.0, #0.5,   # with this probability, the labels for real/fake in discriminator loss are flipped, which is a common technique to stabilize training
    'definition_of_one_label': 1.0,#0.7,
    'label': f'gradient_penalty_wasserstein',
    'logging_dir': 'logs',
    })



    #model = CycleGANPrediscriminatorWrapper.load_from_checkpoint(args_dict.discriminator.discriminator_link, weights_only=False, strict=False)
        # strict=False is very important because we are in fact reading the instance of CycleGANWrapper and loading it into CycleGANPrediscriminatorWrapper
    #model.hparams.args_dict = args_dict # overwritting previously saved args_dict

    #model = CycleGANWrapper.load_from_checkpoint(args_dict.discriminator.discriminator_link, weights_only=False, strict=False)
    #model.hparams.args_dict = args_dict  # overwritting previously saved args_dict

    #donor_synthetic = AutoencoderTwoWayWrapper.load_from_checkpoint(args_dict.generator.synthetic_generator_link, weights_only=False)
    #donor_experimental = AutoencoderTwoWayWrapper.load_from_checkpoint(args_dict.generator.experimental_generator_link, weights_only=False)
    #model.transplant_generator_heads(donor_synthetic, donor_experimental)

    #model.transplant_prediscriminator_heads(donor_synthetic, donor_experimental)
    #model.reinstaintiate_discriminator()  # we need to reinstaintiate the discriminator because of different structure

    model = CycleGANWrapper(args_dict)
    donor_synthetic = AutoencoderTwoWayWrapper.load_from_checkpoint(args_dict.generator.synthetic_generator_link,
                                                                    weights_only=False)
    donor_experimental = AutoencoderTwoWayWrapper.load_from_checkpoint(args_dict.generator.experimental_generator_link,
                                                                       weights_only=False)

    model.transplant_generator_heads(donor_synthetic, donor_experimental)

    dataset = instantiate(args_dict.data.dataset)
    dataloader = DataLoader(dataset, batch_size=args_dict.batch_size, shuffle=True, num_workers=args_dict.num_workers)

    loggers = [pl.loggers.TensorBoardLogger(save_dir='.', name=args_dict.logging_dir, default_hp_metric=False,
                                            version=args_dict.label + "-" + bio_PLB.tools.get_git_revision_short_hash())]

    trainer = pl.Trainer(
        # default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        # We run on a single GPU (if possible)
        deterministic=True,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        # precision='16-mixed',
        # How many epochs to train for if no patience is set
        max_epochs=args_dict.epochs,
        log_every_n_steps=1,
        # val_check_interval=4, # do not check more often than full epoch
        # limit_train_batches=20,
        # gradient_clip_val=1,
        logger=loggers,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="train_final_loss", save_top_k=3,
                filename='best_total_loss_{epoch}-{train_final_loss:.5f}'
            ),  # Save the best checkpoint based on the min loss recorded. Saves only weights and not optimizer
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="train_cycle_identity_synthetic_loss", save_top_k=3,
                filename='best_train_cycle_identity_synthetic_loss_{epoch}-{train_cycle_identity_synthetic_loss:.5f}-{train_final_loss:.5f}'
            ),  # Save the best checkpoint based on the min loss recorded. Saves only weights and not optimizer
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, every_n_epochs=10, save_top_k=-1, monitor="epoch", mode="max",
                filename='epoch_{epoch}-{train_final_loss:.5f}'
            ),
            pl.callbacks.LearningRateMonitor("epoch"),
            # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
            # pl.callbacks.DeviceStatsMonitor(cpu_stats=True), # statystyki procesora itp - bardzo duzo
        ],  # Log learning rate every epoch
        # progress_bar_refresh_rate=1,
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()