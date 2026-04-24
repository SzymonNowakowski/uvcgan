from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import utils as vutils

from bio_PLB.data.global_and_instance_norm import GlobalAndInstanceNorm


class BioSyntheticCoordinator(Dataset):
    """
    The main CycleGAN coordinator for Grayscale Option B.
    - Length is determined by the Real dataset (shorter).
    - Synthetic samples are drawn randomly from the 1M pool.
    """

    # Static variables for debug sampling
    _samples_saved = 0
    _max_samples = 10
    _debug_dir = Path("debug_inputs")

    def __init__(self, synth_adapter, real_dataset, shared_transforms=None):
        self.synth_adapter = synth_adapter
        self.real_dataset = real_dataset
        self.shared_transforms = shared_transforms

        # Create debug directory if it doesn't exist
        if not BioSyntheticCoordinator._debug_dir.exists():
            BioSyntheticCoordinator._debug_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, index):
        # 1. Real Image (Sequential Domain B)
        image_real = self.real_dataset[index]

        # 2. Synthetic Image (Randomly sampled Domain A)
        random_idx = np.random.randint(0, len(self.synth_adapter))
        image_synth = self.synth_adapter[random_idx]

        # Final shared transforms
        if self.shared_transforms is not None:
            for shared_transform in self.shared_transforms:
                image_real = shared_transform(image_real)
                image_synth = shared_transform(image_synth)

        image_real = GlobalAndInstanceNorm(global_mean=0.2363, global_std=0.1224)(image_real)
        image_synth = GlobalAndInstanceNorm(global_mean=0.7367, global_std=0.1922)(image_synth)

        # Save a few samples for visual inspection
        if BioSyntheticCoordinator._samples_saved < BioSyntheticCoordinator._max_samples:
            self._dump_debug_sample(image_synth, image_real)
            BioSyntheticCoordinator._samples_saved += 1

        return image_synth, image_real

    def _dump_debug_sample(self, synth, real):
        """
        Saves a side-by-side comparison of synthetic and real inputs to disk
        and prints range statistics for verification.
        """
        idx = BioSyntheticCoordinator._samples_saved
        save_path = BioSyntheticCoordinator._debug_dir / f"input_check_{idx}.png"

        # Check if file exists to avoid redundant I/O operations across epochs/workers
        if not save_path.exists():
            # Extract scalar values from tensors for logging
            s_min, s_max = synth.min().item(), synth.max().item()
            r_min, r_max = real.min().item(), real.max().item()

            print(f"--- Debug Sample {idx} Stats ---")
            print(f"  Synthetic: min={s_min:.4f}, max={s_max:.4f}")
            print(f"  Real:      min={r_min:.4f}, max={r_max:.4f}")
            print(f"  Saving to: {save_path}")

            # Concatenate tensors along the width dimension (dim=2 for CHW format)
            # to create a single side-by-side comparison image
            grid = torch.cat([synth, real], dim=2)

            # Save the image. normalize=True is crucial as it shifts the input
            # range (e.g., [-1, 1]) to [0, 1] for correct PNG representation.
            vutils.save_image(grid, save_path, normalize=True)
