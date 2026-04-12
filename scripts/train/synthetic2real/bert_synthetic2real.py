import os
import argparse
from uvcgan import ROOT_OUTDIR, ROOT_DATA, train



# Note: Classes are not defined here to keep the entry point clean.
# They are dynamically loaded via hydra.utils.instantiate in uvcgan/data/data.py
# based on the '_target_' paths provided in dataset_args.

def parse_cmdargs():
    parser = argparse.ArgumentParser(description='Pretrain Bio-BERT')
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":


    print("Current Working Directory:", os.getcwd())
    cmdargs = parse_cmdargs()

    args_dict = {
        'batch_size': cmdargs.batch_size,
        'data': {
            'dataset': 'unpaired-bio',  # Triggers the custom logic in uvcgan/data/data.py
            'dataset_args': {
                # 1. Main entry point: The Coordinator that syncs Domain A and B
                '_target_': 'uvcgan.data.datasets.bio_dataset.UnpairedBioCoordinator',

                # 2. First constructor argument: The adapter for synthetic data (Domain A)
                'synth_adapter': {
                    '_target_': 'uvcgan.data.datasets.bio_dataset.SyntheticPLBAdapter',
                    'plb_instance': {
                        # Recursive instantiation of the external research dataset
                        '_target_': 'uvcgan.data.external.PLB.regression.src.plbregression.dataset.PLBDataset',
                        'data_dir': os.path.join(ROOT_DATA,"synthetic2real/synthetic_0.5_px_nm/dataset_01_20260223/"),
                        'return_tensors': False,
                        'transforms': [
                            {
                                '_target_': 'uvcgan.data.external.PLB.regression.src.plbregression.dataset.RandomRotatedShiftedCrop',
                                'size': 160,
                                'interpolation': 'cubic'
                            },
                        # no microscopic noise
                        ],
                    }
                },

                # 3. Second constructor argument: The real biological dataset (Domain B)
                'real_dataset': {
                    '_target_': 'uvcgan.data.datasets.bio_dataset.RealBiologicalDataset',
                    'image_dir': os.path.join(ROOT_DATA,"synthetic2real/real/crop_2957"),
                    'metadata_csv_path': os.path.join(ROOT_DATA,"synthetic2real/real/data_summary_2957.csv"),
                    'target_nm': 320,
                    'target_px': 160
                },

                # Note: 'shared_transform' is injected automatically by 'instantiate'
                # inside uvcgan/data/data.py using 'transform_train/val'
            },
            'transform_train': None,
            'transform_val': None,
        },
        'image_shape': (1, 160, 160),
        'epochs': 1000,
        'discriminator': None,
        'generator': {
            'model' : 'vit-unet',
            'model_args' : {
                'features'           : 384,
                'n_heads'            : 6,
                'n_blocks'           : 12,
                'ffn_features'       : 1536,
                'embed_features'     : 384,
                'activ'              : 'gelu',
                'norm'               : 'layer',
                'unet_features_list' : [48, 96, 192, 384],
                'unet_activ'         : 'leakyrelu',
                'unet_norm'          : 'instance',
                'unet_downsample'    : 'conv',
                'unet_upsample'      : 'upsample-conv',
                'rezero'             : True,
                'activ_output'       : 'sigmoid',
            },
            'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : cmdargs.batch_size * 5e-3 / 512,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        }
    },
    'model'      : 'autoencoder',
    'model_args' : {
        'joint'   : True,
        'masking' : {
            'name'       : 'image-patch-random',
            'patch_size' : (16, 16),
            'fraction'   : 0.4,
        },
    },
    'scheduler' : {
        'name'      : 'CosineAnnealingWarmRestarts',
        'T_0'       : 500,
        'T_mult'    : 1,
        'eta_min'   : cmdargs.batch_size * 5e-8 / 512,
    },
    'loss'             : 'l1',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
# args
    'label'      : f'bert-vit-unet-12-256',
    'outdir'     : os.path.join(ROOT_OUTDIR, 'synthetic2real'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 100,
    }

    train(args_dict)