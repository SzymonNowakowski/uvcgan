import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_blur(img_tensor, kernel_size):
    """
    Python port of the C++ image_manipulation::gaussian_blur.
    Assumes img_tensor shape is (B, H, W).
    """
    # 1. Convert to float for precise convolution math
    img_float = img_tensor.astype(np.float64)

    # 2. Apply the Gaussian Filter
    # sigma: we apply kernel_size to the H and W axes (1 and 2),
    # and 0 to the Batch axis (0) so we don't blur across different images.
    # mode='wrap': This would exactly replicate the C++ 'while(ind_x < 0) ind_x += width' (which is known as Periodic Boundary Conditions)
    # however I have seen ghosting effects at boundaries of the image coming from the artifacts on the other side, that is why I am going to use mode='reflect'

    blurred = gaussian_filter(
        img_float,
        sigma=(0, kernel_size, kernel_size),
        mode='reflect'
    )

    return blurred

def gaussian_noise(img_tensor, magnitude = 50.0):
    """
    Python port of the C++ image_manipulation::gaussian_noise.
    Assumes img_tensor shape is (B, H, W).
    """

    # 1. Thread-safe local generator.
    # default_rng(None) pulls entropy from the OS (like /dev/urandom).
    # This ensures each thread/call is independently seeded without global locks.
    rng = np.random.default_rng()

    # 2. Convert to float to avoid overflow/underflow during arithmetic
    img_float = img_tensor.astype(np.float64)

    # 3. Generate noise using the local generator
    # Mean=0, Std=1/6 matches the C++ std::normal_distribution
    noise = rng.normal(loc=0.0, scale=1.0 / 6.0, size=img_float.shape)

    # 4. Apply magnitude and add to the image
    noisy_img = img_float + (noise * magnitude)

    return noisy_img

def create_grains_batch(batch_size, height, width,
                        grain_size_params,  # (center, width)
                        grain_number_params,  # (center, width)
                        magnitude):
    """
    Python port of the C++ image_manipulation::create_grains
    Creates a batch of unique grain images (B, H, W).
    Each image in the batch has a different number, size, and position of grains.
    """
    # 1. Thread-safe, independently seeded generator
    rng = np.random.default_rng()

    # 2. Initialize the 3D tensor (Batch, Height, Width)
    batch_grains = np.zeros((batch_size, height, width), dtype=np.float64)

    # 3. Iterate through each image in the batch to ensure uniqueness
    for b in range(batch_size):
        # Determine number of grains for THIS specific image
        n = int(rng.normal(grain_number_params[0], grain_number_params[1]))
        n = max(0, n)

        if n == 0:
            continue

        # Generate unique parameters for ALL grains in this image at once
        # Using 'size=n' creates an array of unique values for this specific frame
        sizes = np.abs(rng.normal(grain_size_params[0], grain_size_params[1], size=n)).astype(int)
        cxs = rng.integers(0, width, size=n)
        cys = rng.integers(0, height, size=n)
        intensities = magnitude * rng.normal(loc=0.0, scale=1.0 / 6.0, size=n)

        # 4. Draw each grain into the current batch slice 'b'
        for i in range(n):
            s = sizes[i]
            if s == 0:
                continue

            # Create a local circular coordinate mask
            y_grid, x_grid = np.ogrid[-s: s + 1, -s: s + 1]
            mask = x_grid ** 2 + y_grid ** 2 <= s ** 2

            # Extract coordinates where the mask is True
            y_idx, x_idx = np.where(mask)

            # Calculate global coordinates with Periodic Wrapping (%)
            # This handles grains that 'leak' off one side and appear on the other
            rows = (cys[i] + y_idx - s) % height
            cols = (cxs[i] + x_idx - s) % width

            # Accumulate intensity into the specific image slice
            # np.add.at is used here because one grain might overlap itself
            # if it's larger than the image, or multiple pixels in the mask
            # might map to the same wrapped coordinate.
            np.add.at(batch_grains[b], (rows, cols), intensities[i])

    # 5. Per-image Normalization (Contrast Stretching)
    # This matches the C++ normalize_to_uchar255 logic applied to each frame.
    for b in range(batch_size):
        img = batch_grains[b]
        f_min, f_max = img.min(), img.max()

        if f_max > f_min:
            # Scale the float values to a 0.0 - 255.0 range
            batch_grains[b] = (img - f_min) * (255.0 / (f_max - f_min))
        else:
            # Handle empty/flat images by setting them to neutral gray
            batch_grains[b] = 128.0

    return batch_grains


def interpolate(A, B, strength, clip=True):
    """
    Generalized additive interpolator:
    Z = (1-strength) A + strength B

    Args:
        A (np.ndarray): Base image/tensor (Batch, H, W).
        B (np.ndarray): Secondary signal/tensor to be added.
        strength (float or np.ndarray): Multiplier for the B signal.
                                        Can be a scalar or a per-batch array.
        clip (bool): If True, clips result to [0, 255] and returns uint8.
                     If False, returns raw float64.
    """
    # 1. Promote to float64 for intermediate math to prevent 8-bit overflow
    A_f = A.astype(np.float64)
    B_f = B.astype(np.float64)

    # 2. Additive Model: Signal A + weighted Signal B
    # This treats B as an overlay whose 'impact' is defined by strength.
    Z = (1-strength) * A_f + strength * B_f

    # 3. Finalization
    if clip:
        # Simulates sensor saturation by capping values at 255
        return np.clip(Z, 0, 255).astype(np.uint8)

    return Z

def add_microscopic_noise(img_tensor, strength, grain_size_params = (50, 10), grain_number_params = (50, 1), magnitude = 30.0, grains_blur = 4.0, blur = 2.0, gaussian_noise_magnitude = 50.0):
    """
    This is the MAIN FUNCTION that
    adds microscopic noise to the input image batched tensor.
    Assumes img_tensor shape is (B, H, W).
    """

    # 1. Generate the raw crystal grains
    raw_grains = create_grains_batch(
        batch_size=img_tensor.shape[0],
        height=img_tensor.shape[1],
        width=img_tensor.shape[2],
        grain_size_params=grain_size_params,
        grain_number_params=grain_number_params,
        magnitude=magnitude
    )

    # 2. Apply mutual blur to the grains
    mutual_blur = (grains_blur ** 2 + blur ** 2) ** 0.5
    blurred_grains = gaussian_blur(raw_grains, kernel_size=mutual_blur)

    # 3. Blur the base image and add the grains
    # We use clip=False here to keep the precision
    mixed_signal = interpolate(
        A=gaussian_blur(img_tensor, kernel_size=blur),
        B=blurred_grains,
        strength=0.5,
        clip=False
    )

    # 4. Final Step: Add sensor noise
    max_noise = gaussian_noise(mixed_signal, magnitude=gaussian_noise_magnitude)

    # We use clip=True here to simulate the final output of a camera sensor, which cannot represent values outside the 0-255 range.
    return interpolate(
        A=img_tensor,
        B=max_noise,
        strength=strength,
        clip=True
    )

def add_microscopic_noise_single(image, strength, grain_size_params=(50, 10), grain_number_params=(50, 1), magnitude=30.0, grains_blur=4.0, blur=2.0, gaussian_noise_magnitude=50.0):
    return add_microscopic_noise(image[None], strength, grain_size_params, grain_number_params, magnitude, grains_blur, blur, gaussian_noise_magnitude)[0]