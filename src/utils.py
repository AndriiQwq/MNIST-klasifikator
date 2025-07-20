import torch
import torch.optim as optim
import torch.nn as nn
from colorama import Fore
from model import MLP


def get_device():
    """Get the best available device"""
    torch.manual_seed(1)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f'Using device: {device}')
    return device


def load_or_create_model(load_model, device):
    """Load existing model or create new one"""
    if load_model == False:
        model = MLP()
    else:
        model = torch.load('models/mnist_model.pth')
        print(Fore.YELLOW + 'Model loaded')
    
    # Move model to the device(GPU/CPU)
    model = model.to(device)
    return model


def create_optimizer(model, optimizer_type, learning_rate, momentum):
    """Create optimizer based on configuration"""
    if optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd_momentum':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_loss_function():
    """Get loss function"""
    return nn.CrossEntropyLoss()


def save_model(model, filename='models/mnist_model.pth'):
    """Save model state dict"""
    torch.save(model.state_dict(), filename)
    print(f'Model saved to {filename}')
