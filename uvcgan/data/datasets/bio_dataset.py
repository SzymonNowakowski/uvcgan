import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from data.external.PLB.regression.src.plbregression.dataset import PLBDataset


class BiologicalRealDataset(Dataset):
    """
    Handles real biological images (.jpg) and uses CSV metadata
    for domain-specific preprocessing.
    """

    def __init__(self, image_dir, metadata_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # We will implement the CSV-based scaling/logic here
        # to keep the Hybrid Coordinator clean.
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        file_path = os.path.join(self.image_dir, filename)

        # Load as grayscale (Domain B)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # TODO: Apply CSV metadata adjustments here (resolution/scaling)

        if self.transform:
            image = self.transform(image)
        return image


class SyntheticPLBAdapter(Dataset):
    """
    A lightweight wrapper for the external PLBDataset engine.
    Prepares raw synthetic data for the hybrid pipeline.
    """

    def __init__(self, plb_instance, adapter_transform=None):
        self.plb = plb_instance
        self.adapter_transform = adapter_transform

    def __len__(self):
        return len(self.plb)

    def __getitem__(self, index):
        # Fetch data from the submoduled research code
        image_raw, _ = self.plb[index]

        # Domain A specific prep
        if self.adapter_transform:
            image_raw = self.adapter_transform(image_raw)
        return image_raw


class UnpairedBioCoordinator(Dataset):
    """
    The main CycleGAN coordinator for Grayscale Option B.
    - Length is determined by the Real dataset (shorter).
    - Synthetic samples are drawn randomly from the 1M pool.
    """

    def __init__(self, synth_adapter, real_dataset, shared_transform=None):
        self.synth_adapter = synth_adapter
        self.real_dataset = real_dataset
        self.shared_transform = shared_transform

    def __len__(self):
        # Epoch length matches the biological data count (~3000)
        return len(self.real_dataset)

    def __getitem__(self, index):
        # 1. Real Image (Sequential Domain B)
        image_real = self.real_dataset[index]

        # 2. Synthetic Image (Randomly sampled Domain A)
        random_idx = np.random.randint(0, len(self.synth_adapter))
        image_synth = self.synth_adapter[random_idx]

        # Final shared transforms (Normalization to [-1, 1], Centering, etc.)
        if self.shared_transform:
            image_real = self.shared_transform(image_real)
            image_synth = self.shared_transform(image_synth)

        return image_synth, image_real