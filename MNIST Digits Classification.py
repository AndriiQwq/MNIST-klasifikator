import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

import time
import os
import configparser
from colorama import Fore, init

init(autoreset=True)

config_file = 'config.ini'
config = configparser.ConfigParser()

"""Parameters for the configuration file"""
load_model = None
auto_save_model = None
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
        'auto_save_model': 'False',
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
        load_model, save_model

    load_model = config.getboolean('Settings', 'load_model')
    save_model = config.getboolean('Settings', 'auto_save_model')
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
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)  # output layer
        return x


if load_model == False:
    model = MLP()
else:
    model = torch.load('mnist_model.pth')
    print(Fore.YELLOW + 'Model loaded')

"""Loss function"""
loss_function = nn.CrossEntropyLoss()

"""Activation function"""
if optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # nesterov=True
elif optimizer == 'sgd_momentum':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
elif optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

torch.manual_seed(1)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'Using device: {device}')


def plot_training_losses(train_losses, test_accuracies):
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


def plot_testing_losses(evaluate_loss, evaluate_accuracy):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(evaluate_loss, label='Evaluate Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(evaluate_accuracy, label='Evaluate Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()



def plot_confusion_matrix(model):
    _, _, all_preds, all_labels = evaluate(model)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


def train(model, optimizer, epochs=10):
    global save_model

    best_result = {
        'max_accuracy': 0,
        'eval_loss': 0,
        'epoch': 0,
        'train_loss': 0,
        'train_accuracy': 0
    }

    accepting_rate = 0.97
    training_accuracies = []
    training_losses = []
    evaluation_accuracies = []
    evaluation_losses = []

    """"TTL - Time to live. If model does not update its accuracy by count of lives,
        process of find best model will be terminated"""
    live = 150
    TTL = live

    for epoch in range(epochs):
        model.train()

        """Training information"""
        training_info = {
            'running_loss': 0.0,
            'correct': 0,
            'total': 0
        }

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            """Append training information"""
            training_info['running_loss'] += loss.item()
            _, predicted = torch.max(output, 1)  # predicated output
            training_info['total'] += labels.size(0)
            training_info['correct'] += (predicted == labels).sum().item()

        training_accuracy = training_info['correct'] / training_info['total']
        training_accuracies.append(training_accuracy)
        training_losses.append(training_info['running_loss'] / len(train_loader))

        """Evaluation model"""
        evaluation_accuracy, evaluation_loss, _, _ = evaluate(model)
        evaluation_accuracies.append(evaluation_accuracy)
        evaluation_losses.append(evaluation_loss)

        if evaluation_accuracy > best_result['max_accuracy']:
            best_result['max_accuracy'] = evaluation_accuracy
            best_result['epoch'] = epoch
            best_result['train_loss'] = training_info['running_loss'] / len(train_loader)
            best_result['train_accuracy'] = training_accuracy
            best_result['eval_loss'] = evaluation_loss

            TTL = live
            if save_model and best_result['max_accuracy'] > accepting_rate:
                torch.save(model.state_dict(), 'mnist_model.pth')
        else:
            TTL -= 1

        """Logging"""
        print(Fore.GREEN + "--------------------------------------------------------------------------------\n",
              Fore.MAGENTA + f"Epoch {epoch + 1},",
              Fore.RED + f" Train Loss: {training_info['running_loss'] / len(train_loader):.4f}, ",
              Fore.GREEN + f"Train Accuracy: {training_accuracy:.4f}, ",
              Fore.RED + f"Test Loss: {evaluation_loss:.4f}, ",
              Fore.GREEN + f"Test Accuracy: {evaluation_accuracy:.4f}",
              Fore.GREEN + "\n--------------------------------------------------------------------------------")

        if TTL == 0:
            print(Fore.GREEN + f"Best model was not reached by {live} hope.")
            break

    """Conclusion"""
    print(Fore.LIGHTGREEN_EX + f"\n\n\nBest epoch {best_result['epoch']} with:"
                               f"Train Loss: {best_result['train_loss']:.4f}, "
                               f"Train Accuracy: {best_result['train_accuracy']:.4f}, "
                               f"Test Loss: {best_result['eval_loss']:.4f}, "
                               f"Test Accuracy: {best_result['max_accuracy']:.4f}")

    """Plotting graphs"""
    plot_testing_losses(evaluation_losses, evaluation_accuracies)
    plot_training_losses(training_losses, training_accuracies)

    if show_confusion_matrix:
        plot_confusion_matrix(model)



def evaluate(model):
    model.eval()

    training_info = {
        'correct': 0,
        'total': 0,
        'running_loss': 0.0
    }
    all_preds = []
    all_labels = []

    for input, label in test_loader:
        with torch.no_grad():

            outputs = model(input)
            loss = loss_function(outputs, label)
    
            _, predicted = torch.max(outputs, 1)
            training_info['total'] += label.size(0)
    
            correct_pred = (predicted == label).sum().item()
            training_info['correct'] += correct_pred
    
            training_info['running_loss'] += loss.item()
    
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    evaluation_accuracy = training_info['correct'] / training_info['total']
    evaluation_loss = training_info['running_loss'] / len(test_loader)
    return evaluation_accuracy, evaluation_loss, all_preds, all_labels



if __name__ == '__main__':
    global save_model
    start_time = time.time()

    train(model, optimizer, epoch_count)

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
