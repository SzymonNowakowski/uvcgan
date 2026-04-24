from torch.utils.data import Dataset

from uvcgan.data.external.PLB.regression.src.plbregression.dataset import PLBDataset


class SyntheticDatasetAdapter(Dataset):
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
