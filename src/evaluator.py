import torch


def evaluate(model, test_loader, loss_function, device):
    """Evaluate model on test dataset"""
    model.eval()

    eval_info = {
        'correct': 0,
        'total': 0,
        'running_loss': 0.0
    }
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Transfer inputs and labels to the device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        
            _, predicted = torch.max(outputs, 1)
            eval_info['total'] += labels.size(0)
        
            correct_pred = (predicted == labels).sum().item()
            eval_info['correct'] += correct_pred
        
            eval_info['running_loss'] += loss.item()
        
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    evaluation_accuracy = eval_info['correct'] / eval_info['total']
    evaluation_loss = eval_info['running_loss'] / len(test_loader)
    return evaluation_accuracy, evaluation_loss, all_preds, all_labels
