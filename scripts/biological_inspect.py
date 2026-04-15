import os
import cv2
import torch
import numpy as np
from PIL import Image

# Importing your specific Bio-Normalization and UVCGAN utilities
from uvcgan.utils.model_state import ModelState
from uvcgan.eval.funcs import tensor_to_image, start_model_eval

from uvcgan.data.datasets.bio_dataset import GlobalAndInstanceNorm


def unnormalize_to_hwc(tensor, mean, std):
    """
    Reverses the Global Standardization to bring the tensor
    back to a 0-1 range based on Domain B (Real) statistics.
    """
    # Move to CPU and convert to numpy
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img.squeeze(0)  # Remove batch dim

    # Reverse Global Normalization: (x * std) + mean
    img = (img * std) + mean

    # Transpose from (C, H, W) to (H, W, C)
    img = img.transpose((1, 2, 0))
    return img


def run_bio_inference():
    # --- 1. CONFIGURATION ---
    model_path = 'outdir/synthetic2real/model_d(unpaired-bio)_m(cyclegan)_d(basic)_g(vit-unet)_unpaired-bio_vit-unet-12-self-lsgan-paper-cycle_high-160px'
    input_dir = 'input/images'
    output_dir = 'outputdir/images'
    os.makedirs(output_dir, exist_ok=True)

    # Domain Statistics from your Coordinator code
    SYNTH_STATS = {'mean': 0.7367, 'std': 0.1922}
    REAL_STATS = {'mean': 0.2363, 'std': 0.1224}

    # Load Model (Synthetic -> Real)
    model_state = ModelState.from_str('eval')
    args, model, _ = start_model_eval(model_path, epoch=-1, model_state=model_state)
    model.eval()

    # Normalizer for the Input (Synthetic)
    bio_norm = GlobalAndInstanceNorm(SYNTH_STATS['mean'], SYNTH_STATS['std'])

    # --- 2. PROCESSING ---
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in image_files:
        # Load Raw Synthetic
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # Resize to 160x160
        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)

        # Prep Tensor (0 to 1 range)
        input_tensor = torch.from_numpy(img).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 160, 160)

        # Normalize Input
        input_tensor = bio_norm(input_tensor).to(model.device)

        with torch.no_grad():
            # Create a dummy tensor of the same shape (1, 1, 160, 160)
            dummy_tensor = torch.zeros_like(input_tensor)

            # Pass both to the model as a list
            model.set_input([dummy_tensor, input_tensor])

            model.forward_nograd()
            translated_tensor = model.images.fake_a

        if translated_tensor is not None:
            # --- 3. UNNORMALIZE USING REAL STATS ---
            # This reverses the global shift so the "fake real" image
            # matches the brightness/contrast of the "actual real" images.
            output_hwc = unnormalize_to_hwc(
                translated_tensor[0],
                mean=REAL_STATS['mean'],
                std=REAL_STATS['std']
            )

            # Final scaling to 8-bit integer
            output_final = np.clip(output_hwc * 255, 0, 255).astype('uint8')

            # Save
            res_img = Image.fromarray(np.squeeze(output_final))
            res_img.save(os.path.join(output_dir, fname))
            print(f"Processed and Unnormalized: {fname}")


if __name__ == "__main__":
    run_bio_inference()