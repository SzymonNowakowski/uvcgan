import torch
import numpy as np
from tqdm import tqdm


def run_once_calculate_stats(coordinator, num_samples=2000):
    """
    Perform a one-time calculation of global mean and standard deviation.
    Run this script once, note the values, and hardcode them into your
    GlobalAndInstanceNorm setup.
    """
    print(f"Sampling {num_samples} images to calculate global statistics...")

    synth_values = []
    real_values = []

    # Fix seed for one-time calculation consistency
    np.random.seed(42)
    total_len = len(coordinator)

    for _ in tqdm(range(min(num_samples, total_len))):
        idx = np.random.randint(0, total_len)

        # Access raw data (ensure shared_transform=None in coordinator)
        img_s, img_r = coordinator[idx]

        # Flatten and convert to float32 for precise averaging
        synth_values.append(img_s.flatten().astype(np.float32))
        real_values.append(img_r.flatten().astype(np.float32))

    # Stack all pixels to calculate true global mean and std
    all_synth = np.concatenate(synth_values)
    all_real = np.concatenate(real_values)

    results = {
        "synthetic": {
            "mean": np.mean(all_synth),
            "std": np.std(all_synth)
        },
        "real": {
            "mean": np.mean(all_real),
            "std": np.std(all_real)
        }
    }

    print("\n" + "=" * 40)
    print("FINAL STATISTICS (Range 0-255)")
    print("=" * 40)
    print(f"SYNTHETIC -> Mean: {results['synthetic']['mean']:.4f}, Std: {results['synthetic']['std']:.4f}")
    print(f"REAL      -> Mean: {results['real']['mean']:.4f}, Std: {results['real']['std']:.4f}")
    print("=" * 40)
    print("\nUse these values (divided by 255.0) in your GlobalAndInstanceNorm class.")

    return results

# --- HOW TO USE ---
# 1. Run this function once.
# 2. Copy the printed values.
# 3. In your main training script, use them like this:
#
# REAL_MEAN = <value_from_print> / 255.0
# REAL_STD  = <value_from_print> / 255.0
# custom_norm = GlobalAndInstanceNorm(REAL_MEAN, REAL_STD)