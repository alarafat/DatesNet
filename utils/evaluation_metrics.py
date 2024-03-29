import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def prob2class(predictions, targets):
    pred_prob = torch.exp(predictions)
    pred_probs, pred_classes = torch.max(pred_prob, dim=1)
    gt_probs, gt_classes = torch.max(targets, dim=1)
    return pred_classes.cpu().numpy(), gt_classes.cpu().numpy()


def kl_divergence(predictions, targets):
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    kl_div = criterion(predictions, targets).item()  # Ensure predictions are in log form
    print(f"KL Divergence: {kl_div}")
    return kl_div


def compute_accuracy(predictions, targets):
    pred_classes, gt_classes = prob2class(predictions=predictions, targets=targets)
    accuracy = accuracy_score(gt_classes, pred_classes)
    return accuracy


def compute_metrics(predictions, targets):
    pred_classes, gt_classes = prob2class(predictions=predictions, targets=targets)
    precision, recall, f1, _ = precision_recall_fscore_support(gt_classes, pred_classes, average=None)
    return precision, recall, f1


def compute_confusion_matrix(predictions, targets):
    pred_classes, gt_classes = prob2class(predictions=predictions, targets=targets)
    cm = confusion_matrix(gt_classes, pred_classes, normalize='true')
    return cm
