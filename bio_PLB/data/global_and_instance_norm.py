class GlobalAndInstanceNorm:
    """
    Applies a two-stage normalization pipeline:
    1. Global Standardization using pre-calculated dataset statistics.
    2. Instance Normalization to ensure each sample has zero mean and unit variance.
    """

    def __init__(self, global_mean, global_std):
        # Store dataset-wide statistics calculated offline
        self.global_mean = global_mean
        self.global_std = global_std

    def __call__(self, tensor):
        # 1. Global Standardization
        # Shifts and scales the tensor based on the overall distribution of the source domain.
        # This helps in aligning the dynamic ranges of different datasets.
        tensor = (tensor - self.global_mean) / self.global_std

        # 2. Instance Normalization (Per-sample Z-score)
        # Calculates mean and standard deviation for the current individual sample.
        m = tensor.mean()
        s = tensor.std()

        # Subtract mean and divide by std to center the sample at 0 with a variance of 1.
        # We add a small epsilon (1e-6) to the denominator to prevent division by zero
        # in case of uniform/blank images.
        return (tensor - m) / (s + 1e-6)
