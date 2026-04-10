import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2

from .noise import add_microscopic_noise_single

class PLBDataset(Dataset):
    def __init__(
        self,
        data_dir,
        indices=None,
        transforms=None,
        params_transforms=None,
        return_tensors=False,
        use_params: None | list =None,
        params_means: None | list = None,
        params_stds: None | list = None,
    ):
        """
        PLB Dataset for parameter estimation.
        
        Args:
            data_dir (str): Path to the directory containing .npy files and metadata.json.
            indices (list or np.ndarray, optional): Indices of the data to use.
            transforms (list of callables): list of transformations to apply to the images. Each transform should take and return a numpy array.
            params_transforms (list of callables): list of transformations to apply to the parameters. Each transform should take and return a numpy array.
            return_tensors (bool): Whether to return tensors instead of numpy arrays.
            use_params (list or None): List of parameter names to use. If None, all parameters are used.
        """
        self.data_dir = data_dir
        self.transforms = transforms if transforms is not None else []
        self.params_transforms = params_transforms if params_transforms is not None else []
        self.return_tensors = return_tensors

        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        param_names_all = self.metadata["params_list"]
        if use_params is not None:
            self.param_indices = np.array([param_names_all.index(p) for p in use_params])
        else:
            self.param_indices = np.arange(len(param_names_all))
        self.param_names = [param_names_all[i] for i in self.param_indices]

        self.param_means = np.array(param_means, dtype=np.float32) if params_means is not None else 0
        self.param_stds = np.array(params_stds, dtype=np.float32) if params_stds is not None else 1

        self.n_images_per_file = self.metadata["n_images_per_file"]
        self.n_files = self.metadata["n_files"]
        self.total_images = self.n_images_per_file * self.n_files

        if indices is None:
            self.indices = np.arange(self.total_images)
        elif isinstance(indices, int):
            self.indices = np.arange(indices)
        else:
            self.indices = np.array(indices)
            
        
        # Internal caches for memmap objects
        self._image_mmaps = {}
        self._param_mmaps = {}

    def __len__(self):
        return len(self.indices)

    def _get_mmap(self, file_idx):
        if file_idx not in self._image_mmaps:
            img_path = os.path.join(self.data_dir, f"images_{file_idx}.npy")
            param_path = os.path.join(self.data_dir, f"params_{file_idx}.npy")
            
            self._image_mmaps[file_idx] = np.load(img_path, mmap_mode="r")
            self._param_mmaps[file_idx] = np.load(param_path, mmap_mode="r")
            
        return self._image_mmaps[file_idx], self._param_mmaps[file_idx]
    
    def apply_transforms(self, data, transforms):
        for transform in transforms:
            data = transform(data)
        return data

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        file_idx = real_idx // self.n_images_per_file
        inner_idx = real_idx % self.n_images_per_file

        images_mmap, params_mmap = self._get_mmap(file_idx)
        
        image = np.array(images_mmap[inner_idx])
        params = np.array(params_mmap[inner_idx])[self.param_indices].astype(np.float32)

        image = self.apply_transforms(image, self.transforms)
        params = self.apply_transforms(params, self.params_transforms)

        if self.return_tensors:
            image = torch.from_numpy(image).float().unsqueeze(0) / 255 # add channel dimension
            params = torch.from_numpy(params).float()

        return image, params #torch.from_numpy(params)

def rotated_crop(img, center, size, angle_deg, interpolation=cv2.INTER_LINEAR):
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M[:,2] += np.array([size/2 - center[0], size/2 - center[1]])
    return cv2.warpAffine(img, M, (size, size), flags=interpolation)

INTERPOLATION_TYPES = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

class RandomRotatedShiftedCrop:
    def __init__(self,
        size,
        interpolation="linear",
        ):
        self.size = size
        self.interpolation = INTERPOLATION_TYPES[interpolation]

    def __call__(self, image):
        h, w = image.shape[:2]
        angle = np.random.uniform(-180, 180)
        max_shift_h = int(np.floor((h - np.sqrt(2) * self.size) / 2))
        max_shift_w = int(np.floor((w - np.sqrt(2) * self.size) / 2))
        shift_w = np.random.uniform(-max_shift_w, max_shift_w)
        shift_h = np.random.uniform(-max_shift_h, max_shift_h)
        crop_center = w / 2 - 0.5 + shift_w , h / 2 - 0.5 + shift_h
        return rotated_crop(image, crop_center, self.size, angle, interpolation=self.interpolation)

class ResizeTo:
    def __init__(self, size, interpolation="linear"):
        self.size = size
        self.interpolation = INTERPOLATION_TYPES[interpolation]

    def __call__(self, image):
        return cv2.resize(image, (self.size, self.size), interpolation=self.interpolation)

class ParamsNormalizer:
    def __init__(self, means, stds, indices_to_use=None, number_of_params=None):
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.indices_to_use = np.array(indices_to_use) if indices_to_use is not None else None
        # if indices_to_use is not None:
        #     self.mask = np.zeros(len(number_of_params), dtype=bool)
        #     self.mask[indices_to_use] = True
        # else:
        #     self.mask = None
    def transform(self, params):
        if self.indices_to_use is not None:
            params = params.copy()
            params[self.indices_to_use] = (params[self.indices_to_use] - self.means) / self.stds
            return params
        return (params - self.means) / self.stds

    def __call__(self, params):
        return self.transform(params)
    
    def inverse_transform(self, params):
        if self.indices_to_use is not None:
            params = params.copy()
            params[self.indices_to_use] = params[self.indices_to_use] * self.stds + self.means
            return params
        return params * self.stds + self.means

class HKLToUnitVector:
    def __init__(self, indices_to_use):
        self.hkl_indices = np.array(indices_to_use)
    def __call__(self, params):
        params = params.copy()
        sorted_hkl = np.sort(np.absolute(params[self.hkl_indices])).astype(np.float32)
        params[self.hkl_indices] = sorted_hkl / np.linalg.norm(sorted_hkl)
        return params

# class ToTensor:
#     def __init__(self, add_channel_dim=True, dtype=torch.float32):
#         self.add_channel_dim = add_channel_dim
#         self.dtype = dtype
#     def __call__(self, image):
#         image = torch.from_numpy(image).to(self.dtype)
#         if self.add_channel_dim:
#             image = image.unsqueeze(0)
#         return image

class AlbumentationImageTransformWrapper:
    def __init__(self, albumentation_transform):
        self.albumentation_transform = albumentation_transform
    def __call__(self, image):
        return self.albumentation_transform(image=image)['image']

class MicroscopicNoise:
    def __init__(self, strength):
        self.strength = strength
    def __call__(self, image):
        return add_microscopic_noise_single(image, self.strength)

class RandomMicroscopicNoise:
    def __init__(self, strength_range=(0.0, 1.0)):
        self.strength_range = strength_range
    def __call__(self, image):
        strength = np.random.uniform(*self.strength_range)
        return add_microscopic_noise_single(image, strength)