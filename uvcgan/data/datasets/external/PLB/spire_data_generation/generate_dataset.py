import numpy as np
from pathlib import Path
from itertools import islice
import json
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime
import random
import tqdm

import spirepy_client as spirepy



intended_crop_size_nm = 300.0
nm_per_pixel = 2.0

unit_cell_scale_range_nm = (50,120)
volume_proportion_range_nm = (0.15, 0.5)
slice_position_range_nm = (-unit_cell_scale_range_nm[1], unit_cell_scale_range_nm[1])

concurrency = 24
seed = None
max_consecutive_generation_failures = 200

n_images_per_file = 100_000 #100_000
n_files = 1
dataset_dir = Path(f"/data_hdd/projects/049/data_generation/dataset_02_{datetime.datetime.now().strftime('%Y%m%d')}_N{n_images_per_file*n_files}")
# remove existing dataset if exists, to avoid confusion with old data.
if dataset_dir.exists():
    remove = input(f"Directory {dataset_dir} already exists. Do you want to remove it and generate a new dataset? (y/n): ")
    if remove.lower() != 'y':
        print("Exiting without generating dataset.")
        exit(0)
    import shutil
    shutil.rmtree(dataset_dir)

image_size_nm = np.sqrt(2)*(intended_crop_size_nm + max(unit_cell_scale_range_nm)) # rotated intended_crop_size_nm +  maximal unit cell for position shift
image_size_px = int(image_size_nm / nm_per_pixel)

# parameters to be randomized for each image, and saved in params_array
params_list = ["uc_scale_ab", "channel_vol_prop", "slice_position", "h", "k", "l"]
image_args = {
    "W": image_size_px,
    "structure_type": 1, # diamond
    # "uc_scale_ab": None, # to be set to a random value in unit_cell_scale_range
    # "uc_scale_c": None, # to be set, equal to uc_scale_ab
    # "channel_vol_prop": None, # to be set to a random value in volume_proportion_range
    "slice_height": image_size_nm,
    "slice_width": image_size_nm,
    "slice_thickness": 70.0, # nm
    # "slice_position": 0.0,
    "membrane_distance": 0.0,
    "membrane_thickness": 5.0, # nm
    # "h": None, # to be set randomly
    # "k": None,
    # "l": None,
    "image_depth": 75,
}

fixed_paramters = image_args.copy()
# for randomized_param in params_list:
#     del fixed_paramters[randomized_param]

# FUNCTIONS

def generate_random_hkl(r=128, rng=random.Random()):
    r2 = r**2
    while True:
        h = rng.randint(-r, r)
        k = rng.randint(-r, r)
        l = rng.randint(-r, r)
        if h**2 + k**2 + l**2 <= r2 and (h, k, l) != (0, 0, 0):
            return h, k, l


rngs = [
    random.Random((seed or 0) + i) if seed is not None else random.Random()
    for i in range(concurrency)
]

def batch_generator(
    n: int,
    concurrency: int,
    fixed_paramters: dict,
    max_consecutive_failures: int = max_consecutive_generation_failures,
    ):

    def get_image_params(rng: random.Random, fixed_params=fixed_paramters): # to ważne, żeby ustawić fixed_paramters jako domyślny parametr. Dzięki temu, gdy wywołamy get_image_params w wielu wątkach bez argumentu fixed_params, nie będą sprawdzały słownika tylko kożystały z "zamrożonej" kopii
        h, k, l = generate_random_hkl(rng=rng)
        unit_cell_scale_nm = rng.uniform(*unit_cell_scale_range_nm)
        volume_proportion = rng.uniform(*volume_proportion_range_nm)
        slice_position = rng.uniform(*slice_position_range_nm)

        return {
            "W": fixed_params["W"],
            "structure_type": fixed_params["structure_type"],
            "uc_scale_ab": unit_cell_scale_nm,
            "uc_scale_c": unit_cell_scale_nm,
            "channel_vol_prop": volume_proportion,
            "slice_height": fixed_params["slice_height"],
            "slice_width": fixed_params["slice_width"],
            "slice_thickness": fixed_params["slice_thickness"],
            "slice_position": slice_position,
            "h": h,
            "k": k,
            "l": l,
            "membrane_distance": fixed_params["membrane_distance"],
            "membrane_thickness": fixed_params["membrane_thickness"],
            "image_depth": fixed_params["image_depth"],
        }


    def one(idx: int):
        slot = idx % concurrency
        params = get_image_params(rng=rngs[slot])
        return spirepy.generate_spire_image(**params), params

    successful = 0
    submitted = 0
    consecutive_failures = 0

    ex = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = set()
        for _ in range(min(concurrency, n)):
            futures.add(ex.submit(one, submitted))
            submitted += 1

        while successful < n:
            if not futures:
                raise RuntimeError("No pending generation tasks left before reaching target image count.")

            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for fut in done:
                futures.remove(fut)
                try:
                    result = fut.result()
                    successful += 1
                    consecutive_failures = 0
                    yield result
                except Exception as e:
                    consecutive_failures += 1
                    tqdm.tqdm.write(f"Generation error ({consecutive_failures}/{max_consecutive_failures}): {type(e).__name__}: {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(
                            f"Stopping after {consecutive_failures} consecutive generation failures."
                        ) from e

                if successful < n:
                    futures.add(ex.submit(one, submitted))
                    submitted += 1
    finally:
        ex.shutdown(wait=True, cancel_futures=False)


# RUN

dataset_dir.mkdir(exist_ok=True)

dataset_metadata = {
    "n_images_per_file": n_images_per_file,
    "nm_per_pixel": nm_per_pixel,
    "n_files": n_files,
    "image_size_px": image_size_px,
    "params_list": params_list,
    "fixed_paramters": fixed_paramters,
    "random_params_ranges": {
        "unit_cell_scale_nm": unit_cell_scale_range_nm,
        "volume_proportion": volume_proportion_range_nm,
        "slice_position": slice_position_range_nm,
    },
    "concurrency": concurrency,
    "max_consecutive_generation_failures": max_consecutive_generation_failures,
}

with open(dataset_dir / "metadata.json", "w") as f:
    json.dump(dataset_metadata, f, indent=4)

for i_file in range(n_files):
    images_file = dataset_dir / f"images_{i_file}.npy"
    params_file = dataset_dir / f"params_{i_file}.npy"

    # images_array = np.memmap(images_file, dtype=np.uint8, mode='w+', shape=(n_images_per_file, image_size_px, image_size_px))
    # params_array = np.memmap(params_file, dtype=np.float32, mode='w+', shape=(n_images_per_file, len(params_list)))
    images_array = np.lib.format.open_memmap(images_file, mode='w+', dtype='uint8', shape=(n_images_per_file, image_size_px, image_size_px))
    params_array = np.lib.format.open_memmap(params_file, mode='w+', dtype='float32', shape=(n_images_per_file, len(params_list)))

    for i, (img, params) in tqdm.tqdm(
        enumerate(batch_generator(n_images_per_file, concurrency, fixed_paramters)),
        total=n_images_per_file,
        desc=f"File {i_file+1}/{n_files}",
    ):
        images_array[i] = img
        params_array[i] = [params[key] for key in params_list]

        # # Optional: Periodically flush to disk if N is extremely large
        if i % 10_000 == 0:
            images_array.flush()
            params_array.flush()

    # 3. Final flush and "close" (deleting the object handles the close)
    images_array.flush()
    params_array.flush()