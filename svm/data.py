from torch.utils.data import Dataset, DataLoader
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


class SVMDataset(Dataset):
    def __init__(self, n_samples, cluster_std):
        self.n_samples = n_samples
        self.X, self.y = make_blobs(n_samples=n_samples, centers=2, cluster_std=cluster_std)
        self.y[np.where(self.y==0)] = -1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), self.y[index] 
