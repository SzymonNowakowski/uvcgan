import os

from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
#from torchvision.transforms import ToTensor

from bio_PLB.models.autoencoder_wrapper import AutoencoderWrapper
#from bio_PLB.external.PLB.regression.src.plbregression.coordinator_dataset import SyntheticDatasetAdapter

import pytorch_lightning as pl
import torch

import bio_PLB.tools

def main():


    print("Current Working Directory:", os.getcwd())

    # register a resolver named "eval"
    # this tells OmegaConf: "When you see ${eval: ...}, run it through Python's eval"
    OmegaConf.register_new_resolver("eval", eval)

    args_dict = OmegaConf.create({
        'outdir': 'outdir',
        'batch_size': 128,
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
                        # TODO: add mean/std
                    },
                    {
                        '_target_': 'bio_PLB.external.PLB.regression.src.plbregression.experimental_dataset.ExperimentalDataset',
                        'image_dir': "data/synthetic2real/backgrounds/tla_spireai",
                        'metadata_csv_path': "data/synthetic2real/backgrounds/background_files.csv",
                        'target_nm': "${eval:'2 * ${target_px}'}",
                        'target_px': '${target_px}',
                        'return_tensors': True,
                        # TODO: add mean/std
                    }
                ],
                'main_dataset': 1,  # experimental dataset is main
                #'shared_transforms': [
                #    {'_target_': 'torchvision.transforms.ToTensor'},
                #]
            },
            'transform_train': None,
            'transform_val': None,
        },
        'epochs': 2000,
        'discriminator': None,
        'generator': {
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
            'optimizer'  : {
            '_target_': 'torch.optim.AdamW',  # Define the class path here
            'lr'      : "${eval:'${batch_size} * 2e-3 / 512'}",
            'betas'   : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        }
    },
    'model'      : 'autoencoder',
    'masking' : {
            '_target_' : 'uvcgan.torch.image_masking.ImagePatchRandomMasking',
            'patch_size' : (16, 16),
            'fraction'   : 0.4,
    },
    'scheduler' : None,#{
        #'_target_' : 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts',
        #'T_0'       : 500,
        #'T_mult'    : 1,
        #'eta_min': "${eval:'${batch_size} * 5e-8 / 512'}",
    #},
    'loss'             : {'_target_' : 'torch.nn.L1Loss'},
    'label': f'bert-vit-unet-12-160px',
    'logging_dir': 'logs',
    })



    model = AutoencoderWrapper(args_dict)
    dataset = instantiate(args_dict.data.dataset)
    dataloader = DataLoader(dataset, batch_size=args_dict.batch_size, shuffle=False, num_workers=args_dict.num_workers)

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
                filename='best_loss_{epoch}-{train_final_loss:.5f}'
            ),  # Save the best checkpoint based on the min loss recorded. Saves only weights and not optimizer
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, every_n_epochs=5,
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