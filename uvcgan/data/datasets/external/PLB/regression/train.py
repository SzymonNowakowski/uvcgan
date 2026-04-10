import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import albumentations as A
from omegaconf import OmegaConf
from hydra.utils import instantiate
import timm
from pathlib import Path
import datetime
import matplotlib.pyplot as plt

from plbregression.dataset import PLBDataset, RandomRotatedShiftedCrop, ResizeTo, HKLToUnitVector, ParamsNormalizer, AlbumentationImageTransformWrapper


config = OmegaConf.create({
    "data": {
        "training": {
            "_target_": "plbregression.dataset.PLBDataset",
            "data_dir": "../../data_generation/dataset_02_20260401_N100000",
            "indices": None, #1000
            "use_params": ['uc_scale_ab', 'channel_vol_prop', 'h', 'k', 'l'],
            "return_tensors": True,
            "params_transforms": [
                {
                    "_target_": "plbregression.dataset.ParamsNormalizer",
                    "means": [85.03442   ,  0.32531843],
                    "stds": [20.201132  ,  0.10095225],
                    "indices_to_use": (0,1)
                },
                {
                    "_target_": "plbregression.dataset.HKLToUnitVector",
                    "indices_to_use": (2, 3, 4)
                }
            ],
            "transforms": [
                {
                    "_target_": "plbregression.dataset.RandomRotatedShiftedCrop",
                    "size": 150,
                    "interpolation": "cubic"
                },
                {
                    "_target_": "plbregression.dataset.RandomMicroscopicNoise",
                    "strength_range": (0.0, 1.0)
                },
                ],
        },
    "validation": {
        "noise_0": {
            "_target_": "plbregression.dataset.PLBDataset",
            "data_dir": "../../data_generation/dataset_02_20260401_N10000",
            "indices": 1000, #1000
            "use_params": ['uc_scale_ab', 'channel_vol_prop', 'h', 'k', 'l'],
            "return_tensors": True,
            "params_transforms": [
                {
                    "_target_": "plbregression.dataset.ParamsNormalizer",
                    "means": [85.03442   ,  0.32531843],
                    "stds": [20.201132  ,  0.10095225],
                    "indices_to_use": (0,1)
                },
                {
                    "_target_": "plbregression.dataset.HKLToUnitVector",
                    "indices_to_use": (2, 3, 4)
                }
            ],
            "transforms": [
                {
                    "_target_": "plbregression.dataset.RandomRotatedShiftedCrop",
                    "size": 150,
                    "interpolation": "cubic"
                },
                {
                    "_target_": "plbregression.dataset.MicroscopicNoise",
                    "strength": 0.0
                },
            ],
        },
        "noise_0.5": {
            "_target_": "plbregression.dataset.PLBDataset",
            "data_dir": "../../data_generation/dataset_02_20260401_N10000",
            "indices": 1000, #1000
            "use_params": ['uc_scale_ab', 'channel_vol_prop', 'h', 'k', 'l'],
            "return_tensors": True,
            "params_transforms": [
                {
                    "_target_": "plbregression.dataset.ParamsNormalizer",
                    "means": [85.03442   ,  0.32531843],
                    "stds": [20.201132  ,  0.10095225],
                    "indices_to_use": (0,1)
                },
                {
                    "_target_": "plbregression.dataset.HKLToUnitVector",
                    "indices_to_use": (2, 3, 4)
                }
            ],
            "transforms": [
                {
                    "_target_": "plbregression.dataset.RandomRotatedShiftedCrop",
                    "size": 150,
                    "interpolation": "cubic"
                },
                {
                    "_target_": "plbregression.dataset.MicroscopicNoise",
                    "strength": 0.5
                },
            ],
        },
        "noise_1.0": {
            "_target_": "plbregression.dataset.PLBDataset",
            "data_dir": "../../data_generation/dataset_02_20260401_N10000",
            "indices": 1000, #1000
            "use_params": ['uc_scale_ab', 'channel_vol_prop', 'h', 'k', 'l'],
            "return_tensors": True,
            "params_transforms": [
                {
                    "_target_": "plbregression.dataset.ParamsNormalizer",
                    "means": [85.03442   ,  0.32531843],
                    "stds": [20.201132  ,  0.10095225],
                    "indices_to_use": (0,1)
                },
                {
                    "_target_": "plbregression.dataset.HKLToUnitVector",
                    "indices_to_use": (2, 3, 4)
                }
            ],
            "transforms": [
                {
                    "_target_": "plbregression.dataset.RandomRotatedShiftedCrop",
                    "size": 150,
                    "interpolation": "cubic"
                },
                {
                    "_target_": "plbregression.dataset.MicroscopicNoise",
                    "strength": 1.0
                },
            ],
        }

    }
    },
    "training": {
        "batch_size": 100,
        "num_workers": 8,
        "num_epochs": 5,
        "learning_rate": 1e-3,
        "device": "cuda"
    },
    "model": {
        "_target_": "plbregression.model.TimmWrapper",
        "backbone_name": "resnet18",
        "pretrained": True,
        "in_chans": 1,
        "num_classes": 5,
        "preprocessor": {
            "_target_": "torch.nn.Sequential",
            "_args_": [
                {"_target_": "torch.nn.InstanceNorm2d", "num_features": 1},
                {"_target_": "torchvision.transforms.Resize", "size": 224}
            ]
        },
    },
    "criterion": {
        "weight_direction": 1.0,
        "weight_parameters": 1.0

    },
    "logging": {
        "frequency_batches": 100,
        "checkpoint_frequency_batches": 100, # every 100 batches = every 10_000 samples, 10 per epoch
        "checkpoint_frequency_epochs": None,
        # "checkpoint_frequency_batches": 10_000,
        "run_dir": f"runs/exp_01_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
})

dataset_training = instantiate(config.data.training)
dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers)

datasets_validation = {name: instantiate(val_config) for name, val_config in config.data.validation.items()}
dataloaders_validation = {name: torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers) for name, dataset in datasets_validation.items()}

model = instantiate(config.model).train().to(config.training.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
run_dir = Path(config.logging.run_dir)
run_dir.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config, run_dir / "config.yaml")


def validate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        residuals_direction = []
        residuals_params = []
        for images, params in dataloader:
            images = images.to(config.training.device)
            params = params.to(config.training.device)

            outputs = model(images)

            residuals_params.extend((outputs[:, :2] - params[:, :2]).tolist())
            residuals_direction.extend(torch.rad2deg(torch.arccos(torch.nn.functional.cosine_similarity(outputs[:, 2:], params[:, 2:], dim=1))).tolist())
    
    rmse_direction = np.sqrt(np.mean(np.square(residuals_direction)))
    rmse_parameters = np.sqrt(np.mean(np.square(residuals_params), axis=0))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(residuals_direction, bins=20)
    axs[0].set_title(f"Residuals Direction, RMSE: {rmse_direction:.4f}")
    axs[1].hist(residuals_params, bins=20)
    axs[1].set_title(f"Residuals Parameters, RMSE: {rmse_parameters:.4f}")
    plt.tight_layout()

    return rmse_direction, rmse_parameters, fig


print("Epoch\tBatch\tLoss\tLossDirection\tLossParameters")
for i_epoch in range(config.training.num_epochs):
    model.train()
    for i_batch, (images, params) in enumerate(dataloader_training):
        images = images.to(config.training.device)
        params = params.to(config.training.device)

        outputs = model(images)
        loss_direction = torch.nn.functional.cosine_embedding_loss(outputs[:, 2:], params[:, 2:], torch.ones(outputs.shape[0], device=outputs.device)) * config.criterion.weight_direction
        loss_parameters = torch.nn.functional.mse_loss(outputs[:, :2], params[:, :2]) * config.criterion.weight_parameters
        loss = loss_direction + loss_parameters

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % config.logging.frequency_batches == 0:
            print(f"{i_epoch}\t{i_batch}\t{loss.item():.4f}\t{loss_direction.item():.4f}\t{loss_parameters.item():.4f}")
        if config.logging.checkpoint_frequency_batches is not None and (i_batch % config.logging.checkpoint_frequency_batches == 0):
            torch.save(model.state_dict(), run_dir / f"checkpoint_epoch_{i_epoch}_batch_{i_batch}.pth")
            for validation_name in dataloaders_validation.keys():
                val_rmse_direction, val_rmse_parameters, fig = validate_model(model, dataloaders_validation[validation_name])
                fig.suptitle(f"Validation - {validation_name} - Epoch {i_epoch} Batch {i_batch}\nRMSE Direction: {val_rmse_direction:.4f}, RMSE Parameters: {val_rmse_parameters:.4f}")
                fig.savefig(run_dir / f"validation_{validation_name}_epoch_{i_epoch}_batch_{i_batch}.png")
                plt.close(fig)

    if config.logging.checkpoint_frequency_epochs is not None and (i_epoch % config.logging.checkpoint_frequency_epochs == 0):
        torch.save(model.state_dict(), run_dir / f"checkpoint_epoch_{i_epoch}.pth")