import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size):
        self.fake = np.where(labels == 0)[0]
        self.real = np.where(labels == 1)[0]
        self.half = batch_size // 2
        self.n_batches = min(len(self.fake), len(self.real)) // self.half

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        f, r = np.random.permutation(self.fake), np.random.permutation(self.real)
        for b in range(self.n_batches):
            s, e = b * self.half, (b + 1) * self.half
            yield np.random.permutation(np.concatenate([f[s:e], r[s:e]])).tolist()

def get_dataloaders(task_path, batch_size=32, train_ratio=0.8, holdout=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),    
        ])
    
    dataset = datasets.ImageFolder(root=task_path, transform=transform)
    
    # Extract the numerical labels (0 for fake, 1 for real based on alphabetical folder names)
    targets = np.array(dataset.targets)
    
    # Isolate the exact indices for real and fake images
    fake_indices = np.where(targets == 0)[0]
    real_indices = np.where(targets == 1)[0]
    
    # Shuffle the indices with a fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(fake_indices)
    np.random.shuffle(real_indices)
    
    # Calculate the exact 80% cutoff for both classes independently
    fake_train_size = int(len(fake_indices) * train_ratio)
    real_train_size = int(len(real_indices) * train_ratio)
    
    # Construct the perfectly balanced Train and Test index lists
    train_indices = np.concatenate([fake_indices[:fake_train_size], real_indices[:real_train_size]])
    test_indices = np.concatenate([fake_indices[fake_train_size:], real_indices[real_train_size:]])

    if holdout:   # for hyperparameter search: carve a validation set out of TRAIN
        tr = np.random.RandomState(0).permutation(train_indices)
        cut = int(0.8 * len(tr))
        train_indices, test_indices = tr[:cut], tr[cut:]
    
    # Create the PyTorch Subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Generate the separate loaders
    train_labels = targets[train_indices]   # labels in Subset order
    
    train_loader = DataLoader(train_dataset, batch_sampler=BalancedBatchSampler(train_labels, batch_size), num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
