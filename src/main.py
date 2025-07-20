import time
from colorama import Fore, init

from config_manager import ConfigManager
from data_loader import load_datasets, create_data_loaders
from utils import get_device, load_or_create_model, create_optimizer, get_loss_function, save_model
from training import train

init(autoreset=True)


def main():
    # Initialize configuration
    config_manager = ConfigManager()
    # config_manager.load_config()
    
    # Get device
    device = get_device()
    
    # Load datasets
    train_dataset, test_dataset = load_datasets()
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dataset, 
        test_dataset,
        config_manager.train_batch_size,
        config_manager.test_batch_size
    )
    
    # Create or load model
    model = load_or_create_model(
        config_manager.load_model,
        device
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        config_manager.optimizer,
        config_manager.learning_rate,
        config_manager.momentum
    )
    
    # Get loss function
    loss_function = get_loss_function()
    
    # Training
    print(Fore.CYAN + 'Starting training...\n')
    start_time = time.time()
    
    best_result = train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_function=loss_function,
        device=device,
        config_manager=config_manager
    )
    
    end_time = time.time()
    print(Fore.MAGENTA + f'Training time: {end_time - start_time:.2f} seconds')
    
    print(Fore.GREEN + f'Best model found: '
        f"Epoch: {best_result['epoch']}, "
        f"Train Loss: {best_result['train_loss']:.4f}, "
        f"Train Accuracy: {best_result['train_accuracy']:.4f}, "
        f"Test Loss: {best_result['eval_loss']:.4f}, "
        f"Test Accuracy: {best_result['max_accuracy']:.4f}"
    )
                
    # Ask user to save model
    command = input('Save model? (y/n): ')
    
    if command.lower() == 'y':
        save_model(model)


if __name__ == "__main__":
    main()


