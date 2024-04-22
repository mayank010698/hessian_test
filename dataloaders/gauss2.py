import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class Gauss2(Dataset):
    """Gauss2 two classes"""

    def __init__(self, dataset,  transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample









