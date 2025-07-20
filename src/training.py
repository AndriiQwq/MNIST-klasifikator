import torch
from colorama import Fore, init
from evaluator import evaluate
from visualizer import plot_training_results, plot_confusion_matrix

init(autoreset=True)


def train(model, optimizer, train_loader, test_loader, loss_function, device, config_manager):
    """Train the model"""
    
    # Get config parameters
    epochs = config_manager.epoch_count
    early_stopping_patience = config_manager.early_stopping_patience
    accepting_accuracy_rate = config_manager.accepting_accuracy_rate
    auto_save = config_manager.auto_save_model
    show_confusion_matrix = config_manager.show_confusion_matrix
    
    best_result = {
        'max_accuracy': 0,
        'eval_loss': 0,
        'epoch': 0,
        'train_loss': 0,
        'train_accuracy': 0
    }

    # Use parameters from config
    training_accuracies = []
    training_losses = []
    evaluation_accuracies = []
    evaluation_losses = []

    """"TTL - Time to live. If model does not update its accuracy by count of lives,
        process of find best model will be terminated"""
    TTL = early_stopping_patience

    for epoch in range(epochs):
        model.train()

        """Training information"""
        training_info = {
            'running_loss': 0.0,
            'correct': 0,
            'total': 0
        }

        for inputs, labels in train_loader:
            # Transfer inputs and labels to the device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
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
        evaluation_accuracy, evaluation_loss, _, _ = evaluate(model, test_loader, loss_function, device)
        evaluation_accuracies.append(evaluation_accuracy)
        evaluation_losses.append(evaluation_loss)

        if evaluation_accuracy > best_result['max_accuracy']:
            best_result['max_accuracy'] = evaluation_accuracy
            best_result['epoch'] = epoch
            best_result['train_loss'] = training_info['running_loss'] / len(train_loader)
            best_result['train_accuracy'] = training_accuracy
            best_result['eval_loss'] = evaluation_loss

            TTL = early_stopping_patience
            if auto_save and best_result['max_accuracy'] > accepting_accuracy_rate:
                torch.save(model.state_dict(), 'models/mnist_model.pth')
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
            print(Fore.GREEN + f"Best model was not reached by {early_stopping_patience} hope.")
            break

    """Conclusion"""
    print(Fore.LIGHTGREEN_EX + f"\n\n\nBest epoch {best_result['epoch']} with:"
                               f"Train Loss: {best_result['train_loss']:.4f}, "
                               f"Train Accuracy: {best_result['train_accuracy']:.4f}, "
                               f"Test Loss: {best_result['eval_loss']:.4f}, "
                               f"Test Accuracy: {best_result['max_accuracy']:.4f}")
    
    """Plotting graphs"""
    plot_training_results(training_losses, training_accuracies, evaluation_losses, evaluation_accuracies)
    
    if show_confusion_matrix:
        plot_confusion_matrix(model, test_loader, device)
    
    return best_result
