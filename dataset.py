import os

import cv2 as cv
import numpy as np

ASSETS_PATH = "./archive/assets"
target_classes = [1,2,3,4,5,6,7,8,9]

# Source: 
# https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset?resource=download

def get_sample_names():
    names = []
    for tc in target_classes:
        names.extend([str(tc) + "-" + name for name in os.listdir(f"{ASSETS_PATH}/{tc}/")])
    return names

class Dataset:
    
    def __init__(self, transform=None) -> None:
        self.sample_names = get_sample_names()
        self.num_samples = len(self.sample_names)
        self.transform = transform
        
    def _load_image(self, path):
        img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY).astype(np.float32)
        img = img / 255.
        return img
        
    def __getitem__(self, key):
        if key >= self.num_samples:
            raise IndexError(f"DataLoader only has {self.num_samples} samples, but key index {key} was given.")
        target, name = self.sample_names[key].split("-")
        path = f"{ASSETS_PATH}/{target}/{name}"
        sample = self._load_image(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, int(target)-1
    
    def __len__(self):
        return self.num_samples
        
    
class TrainDataset(Dataset):
    
    def __getitem__(self, key):
        if key >= len(self):
            raise IndexError(f"DataLoader only has {len(self)} samples, but key index {key} was given.")
        return super().__getitem__(2*key)
    
    def __len__(self):
        return super().__len__() // 2
    
class TestDataset(Dataset):
    
    def __getitem__(self, key):
        if key >= len(self):
            raise IndexError(f"DataLoader only has {len(self)} samples, but key index {key} was given.")
        return super().__getitem__(2*key+1)
    
    def __len__(self):
        return super().__len__() // 2