import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from evaluator import evaluate


def plot_training_results(train_losses, train_accuracies, eval_losses, eval_accuracies):
    """Plot training and evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training accuracy
    axes[0, 1].plot(train_accuracies, label='Train Accuracy', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Evaluation loss
    axes[1, 0].plot(eval_losses, label='Eval Loss', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Evaluation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Evaluation accuracy
    axes[1, 1].plot(eval_accuracies, label='Eval Accuracy', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Evaluation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_loader, device):
    """Plot confusion matrix"""
    import torch.nn as nn
    loss_function = nn.CrossEntropyLoss()
    _, _, all_preds, all_labels = evaluate(model, test_loader, loss_function, device)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
