from torch.utils.data import DataLoader, Dataset
import os
import torch
import numpy as np


class ChangeDetectionDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]