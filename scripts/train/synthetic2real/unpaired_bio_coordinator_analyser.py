
import numpy as np
from tqdm import tqdm
import os
from uvcgan import ROOT_DATA
from torchvision import transforms

from uvcgan.data.datasets.bio_dataset import UnpairedBioCoordinator, SyntheticPLBAdapter, RealBiologicalDataset
from uvcgan.data.external.PLB.regression.src.plbregression.dataset import PLBDataset, RandomRotatedShiftedCrop

def run_once_calculate_stats(coordinator):
    """
    Perform a one-time calculation of global mean and standard deviation.
    Run this script once, note the values, and hardcode them into your
    GlobalAndInstanceNorm setup.
    """
    print(f"Sampling {len(coordinator)} images to calculate global statistics...")

    synth_values = []
    real_values = []

    # Fix seed for one-time calculation consistency
    np.random.seed(0)
    total_len = len(coordinator)

    for _ in tqdm(range(total_len), desc="Calculating stats", unit="sample"):
        idx = np.random.randint(0, total_len)

        # Access raw data (ensure shared_transform=None in coordinator)
        img_s, img_r = coordinator[idx]

        # Flatten and convert to float32 for precise averaging
        synth_values.append(img_s.flatten())
        real_values.append(img_r.flatten())

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





# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Initializing components for statistics calculation...")

    # 1. Initialize the Synthetic Domain A (Adapter + Underlying PLB Dataset)
    # We apply the crop but NO microscopic noise or tensor conversion
    plb_internal = PLBDataset(
        data_dir=os.path.join(ROOT_DATA, "synthetic2real/synthetic_0.5_px_nm/dataset_01_20260223/"),
        return_tensors=False,
        transforms=[
            RandomRotatedShiftedCrop(size=160, interpolation='cubic')
        ]
    )

    synth_adapter = SyntheticPLBAdapter(plb_instance=plb_internal)

    # 2. Initialize the Real Biological Domain B
    real_dataset = RealBiologicalDataset(
        image_dir=os.path.join(ROOT_DATA, "synthetic2real/real/crop_2957"),
        metadata_csv_path=os.path.join(ROOT_DATA, "synthetic2real/real/data_summary_2957.csv"),
        target_nm=320,
        target_px=160
    )

    # 3. Initialize the Coordinator with shared_transform=None
    # This ensures we get the raw numpy/uint8 arrays in the range [0, 255]
    coordinator = UnpairedBioCoordinator(
        synth_adapter=synth_adapter,
        real_dataset=real_dataset,
        shared_transform=transforms.ToTensor()
    )

    # 4. Run the calculation
    # This will iterate through the data and print your Mean and Std
    stats = run_once_calculate_stats(coordinator)


# --- HOW TO USE ---
# 1. Run this function once with GlobalAndInstanceNorm turned off (e.g. commented out).
# 2. Copy the printed values.
# 3. In your main training script, use them like this:
#
# REAL_MEAN = <value_from_print>
# REAL_STD  = <value_from_print>
# custom_norm = GlobalAndInstanceNorm(REAL_MEAN, REAL_STD)