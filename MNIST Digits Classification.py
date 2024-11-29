import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import time
import os
import configparser
from colorama import Fore, Style, init

init(autoreset=True)

config_file = 'config.ini'
config = configparser.ConfigParser()

"""Parameters for the configuration file"""
load_model = False
optimizer = None
epoch_count = 0
train_batch_size = 0
test_batch_size = 0
show_confusion_matrix = False
learning_rate = 0
momentum = 0

default_config = {
    'Settings': {
        'load_model': 'False',
        'optimizer(sgd, sgd_momentum, adam)': 'adam',
        'epoch_count': '20',
        'train_batch_size': '64',
        'test_batch_size': '64',
        'show_confusion_matrix': 'False',
        'learning_rate': '0.01',
        'momentum': '0.9',
    }
}


def get_config():
    global optimizer, epoch_count, train_batch_size, test_batch_size, show_confusion_matrix, learning_rate, momentum, \
        load_model

    load_model = config.getboolean('Settings', 'load_model')
    optimizer = config.get('Settings', 'optimizer(sgd, sgd_momentum, adam)')
    epoch_count = config.getint('Settings', 'epoch_count')
    train_batch_size = config.getint('Settings', 'train_batch_size')
    test_batch_size = config.getint('Settings', 'test_batch_size')
    show_confusion_matrix = config.getboolean('Settings', 'show_confusion_matrix')
    learning_rate = config.getfloat('Settings', 'learning_rate')
    momentum = config.getfloat('Settings', 'momentum')


if not os.path.exists(config_file):
    config.read_dict(default_config)
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    print(Fore.LIGHTYELLOW_EX + 'Default configuration was created\n')
    get_config()
else:
    print(Fore.LIGHTYELLOW_EX + 'Configuration configured from config file\n')
    config.read(config_file)
    get_config()

"""Transforms with normalization"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

"""Datasets"""
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

"""DataLoaders"""
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.fc1 = nn.Linear(28 * 28, 128)  # Первый слой
        # self.fc2 = nn.Linear(128, 64)  # Второй слой
        # self.fc3 = nn.Linear(64, 10)  # Выходной слой

        # self.fc1 = nn.Linear(28 * 28, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 10)

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):  # ReLU, Sigmoid, Tanh
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        #x = torch.tanh(self.fc2(x))

        # x = torch.relu(self.fc1(x))  # Активация ReLU
        # x = torch.relu(self.fc2(x))  # Активация ReLU

        #x = self.fc4(x)  # Выходной слой
        return x


model = MLP()

"""Loss function"""
criterion = nn.CrossEntropyLoss()

"""Optimizers"""
if optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # nesterov=True
elif optimizer == 'sgd_momentum':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
elif optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10):
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        accuracy = correct / total
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(Fore.LIGHTMAGENTA_EX + f"Epoch {epoch + 1}/{epochs},",
              Fore.LIGHTRED_EX + f" Train Loss: {running_loss / len(train_loader):.4f}, ",
              Fore.LIGHTGREEN_EX + f"Train Accuracy: {accuracy:.4f}, ",
              Fore.LIGHTCYAN_EX + f"Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_accuracies


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def plot_metrics(train_losses, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def plot_confusion_matrix(model, test_loader):
    """Confusion matrix"""
    y_true = []
    y_pred = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == '__main__':
    global save_model
    start_time = time.time()

    train_losses, test_accuracies = train_model(model, criterion, optimizer, train_loader, test_loader,
                                                epoch_count)

    end_time = time.time()

    print(Fore.MAGENTA + f'Training time: {end_time - start_time:.2f} seconds')

    command = input('Save model? (y/n): ')

    if command == 'y':
        save_model = True
    else:
        save_model = False

    if save_model:
        torch.save(model.state_dict(), 'mnist_model.pth')
        print('Model saved')

    plot_metrics(train_losses, test_accuracies)

    if show_confusion_matrix:
        plot_confusion_matrix(model, test_loader)
