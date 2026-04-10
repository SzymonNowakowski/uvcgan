
from uvcgan.data.external.PLB.regression.src.plbregression.dataset import PLBDataset

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.utils as vutils
from pathlib import Path




class RealBiologicalDataset(Dataset):
    """
    Handles Domain B: Real images with strict geometry and heavy augmentation.
    Combines safe continuous rotation with 90-degree steps and flips.
    """

    def __init__(self, image_dir, metadata_csv_path, target_nm=300, target_px=150):
        self.image_dir = image_dir
        self.target_nm = target_nm
        self.target_px = target_px

        # Load metadata
        df = pd.read_csv(metadata_csv_path, sep=None, engine='python')
        # We keep only rows where invalid_size is NOT 1
        df = df[df['invalid_size'] != 1].reset_index(drop=True)

        self.filenames = df['name_PLB'].values
        self.resolutions = df['resolution_(px/nm)'].values

    def __len__(self):
        return len(self.filenames)

    def _get_safe_params(self, img_w, img_h, crop_size_px):
        """
        Calculates safety boundaries for continuous rotation.
        """
        if img_w < crop_size_px or img_h < crop_size_px:
            raise ValueError(f"Image too small: {img_w}x{img_h}px for {crop_size_px}px crop.")

        r = crop_size_px / 2
        max_r_allowed = min(img_w / 2, img_h / 2)
        full_rot_r = r * 1.4142

        if max_r_allowed >= full_rot_r:
            angle_limit = 45.0  # We only need +/- 45 if we use 90-deg steps
            safe_margin = int(full_rot_r) + 1
        else:
            # Calculate max safe angle to prevent corners from leaving the image
            angle_limit = max(0.0, np.degrees(np.arcsin(max_r_allowed / full_rot_r)) - 45.0)
            safe_margin = int(max_r_allowed)

        return angle_limit, safe_margin

    def __getitem__(self, index):
        filename = self.filenames[index]
        res_px_nm = self.resolutions[index]
        path = os.path.join(self.image_dir, filename)

        crop_size_px = int(self.target_nm * res_px_nm)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None: raise FileNotFoundError(f"Missing: {path}")

        # --- 1. Discrete Augmentation (Flips and 90-deg steps) ---
        # Random Horizontal Flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        # Random Vertical Flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 0)

        # Random 90-degree steps (0, 1, 2, or 3 times 90 deg)
        k = np.random.randint(0, 4)
        img = np.rot90(img, k)
        h, w = img.shape  # compute dimensions after rotation

        # --- 2. Safe Continuous Rotation & Cropping ---
        try:
            angle_limit, margin = self._get_safe_params(w, h, crop_size_px)
        except ValueError as e:
            print(f"CRITICAL: {filename} - {str(e)}")
            raise e

        # Fine-grained rotation within safe limits
        fine_angle = np.random.uniform(-angle_limit, angle_limit)

        # Center-biased sampling with safety margins
        std_x, std_y = w / 8, h / 8
        cx = int(np.clip(np.random.normal(w // 2, std_x), margin, w - margin))
        cy = int(np.clip(np.random.normal(h // 2, std_y), margin, h - margin))

        # Rotation matrix and warp
        matrix = cv2.getRotationMatrix2D((cx, cy), fine_angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)

        # Crop the square
        half_s = crop_size_px // 2
        crop = rotated[cy - half_s: cy - half_s + crop_size_px,
               cx - half_s: cx - half_s + crop_size_px]

        # --- 3. Resize to target_px x target_px ---
        final_img = cv2.resize(crop, (self.target_px, self.target_px), interpolation=cv2.INTER_AREA)

        return final_img.astype(np.uint8)  # Return as uint8 for consistency with PLB data

class SyntheticPLBAdapter(Dataset):
    """
    A lightweight wrapper for the external PLBDataset engine.
    Prepares raw synthetic data for the hybrid pipeline.
    """

    def __init__(self, plb_instance: PLBDataset, adapter_transform=None):
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

    # Static variables for debug sampling
    _samples_saved = 0
    _max_samples = 10
    _debug_dir = Path("debug_inputs")

    def __init__(self, synth_adapter, real_dataset, shared_transform=None):
        self.synth_adapter = synth_adapter
        self.real_dataset = real_dataset
        self.shared_transform = shared_transform

        # Create debug directory if it doesn't exist
        if not UnpairedBioCoordinator._debug_dir.exists():
            UnpairedBioCoordinator._debug_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, index):
        # 1. Real Image (Sequential Domain B)
        image_real = self.real_dataset[index]

        # 2. Synthetic Image (Randomly sampled Domain A)
        random_idx = np.random.randint(0, len(self.synth_adapter))
        image_synth = self.synth_adapter[random_idx]

        # Final shared transforms
        if self.shared_transform:
            image_real = self.shared_transform(image_real)
            image_synth = self.shared_transform(image_synth)

        # Save a few samples for visual inspection
        if UnpairedBioCoordinator._samples_saved < UnpairedBioCoordinator._max_samples:
            self._dump_debug_sample(image_synth, image_real)
            UnpairedBioCoordinator._samples_saved += 1

        return image_synth, image_real

    def _dump_debug_sample(self, synth, real):
        """
        Saves a side-by-side comparison of synthetic and real inputs.
        """
        idx = UnpairedBioCoordinator._samples_saved
        # Concatenate horizontally (dim=2 for CxHxW tensors)
        grid = torch.cat([synth, real], dim=2)

        save_path = UnpairedBioCoordinator._debug_dir / f"input_check_{idx}.png"
        print(f"Saving debug sample {idx} to {save_path}")

        # normalize=True scales values to [0, 1] for correct PNG visualization
        vutils.save_image(grid, save_path, normalize=True)