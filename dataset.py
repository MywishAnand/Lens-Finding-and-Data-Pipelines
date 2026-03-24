import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LensDataset(Dataset):
    def __init__(self, lenses_dir, nonlenses_dir, transform=None):
        self.lenses_dir = lenses_dir
        self.nonlenses_dir = nonlenses_dir
        self.transform = transform
        
        # Load file paths
        self.lenses_files = [os.path.join(lenses_dir, f) for f in os.listdir(lenses_dir) if f.endswith('.npy')]
        self.nonlenses_files = [os.path.join(nonlenses_dir, f) for f in os.listdir(nonlenses_dir) if f.endswith('.npy')]
        
        self.all_files = self.lenses_files + self.nonlenses_files
        self.labels = [1] * len(self.lenses_files) + [0] * len(self.nonlenses_files)
        
    def __len__(self):
        return len(self.all_files)
        
    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        label = self.labels[idx]
        
        # Load numpy array - Expected shape (3, 64, 64)
        image = np.load(file_path)
        
        # Convert to float tensor
        tensor_img = torch.from_numpy(image).float()
        
        if self.transform:
            tensor_img = self.transform(tensor_img)
            
        return tensor_img, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(train_lenses_dir, train_nonlenses_dir, test_lenses_dir, test_nonlenses_dir, batch_size=32, train_transform=None, test_transform=None):
    train_dataset = LensDataset(train_lenses_dir, train_nonlenses_dir, transform=train_transform)
    test_dataset = LensDataset(test_lenses_dir, test_nonlenses_dir, transform=test_transform)
    
    # Calculate positive weight to address class imbalance
    num_neg = len(train_dataset.nonlenses_files)
    num_pos = len(train_dataset.lenses_files)
    pos_weight = float(num_neg) / num_pos if num_pos > 0 else 1.0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, pos_weight
