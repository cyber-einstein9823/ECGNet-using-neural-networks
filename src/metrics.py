import torch
def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()
def confusion_matrix(y_true, y_pred, num_classes=5):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm
def per_class_accuracy(cm):
    return (torch.diag(cm).float() / cm.sum(dim=1).float()).tolist()
