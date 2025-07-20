
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_transforms():
    """Transforms with normalization"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])


def load_datasets():
    """Load MNIST datasets"""
    transform = get_transforms()
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, train_batch_size, test_batch_size):
    """Create DataLoaders"""
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader