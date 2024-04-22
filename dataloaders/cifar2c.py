from torch.utils.data import Dataset, DataLoader
import torch

class Cifar2class(Dataset):
    """Cifar-10 two classes"""

    def __init__(self, dataset,  transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx]