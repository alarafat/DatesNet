import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config.datesnet_config import Config as cfg

from lib.datesnet_mlp import DatesNet
from data.data_processing import DataProcessing
from data.dataset_loader import FERPlusDatasetLoader


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


def compute_metrics(predictions, targets):
    pred_classes, gt_classes = prob2class(predictions=predictions, targets=targets)
    accuracy = accuracy_score(gt_classes, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(gt_classes, pred_classes, average=None)
    return accuracy, precision, recall, f1


def plot_confusion_matrix(predictions, targets, class_names):
    pred_classes, gt_classes = prob2class(predictions=predictions, targets=targets)
    cm = confusion_matrix(gt_classes, pred_classes, normalize='true')
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm.astype(float), annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(cfg.output_dir, r'confusion_matrix_prob.png'))
    plt.show(block=True)


def dump_into_file(kl_div, accuracy, precision, recall, f1, class_names, file_path=r'evaluation_metrics.txt'):
    with open(os.path.join(cfg.output_dir, file_path), 'w') as f:
        f.write("Evaluation Metrics:\n")
        print(f"KL Divergence: {kl_div:.5f}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.5f}%\n\n")
        f.write(f"{'Class':<20}{'Precision':<11}{'Recall':<11}{'F1-Score':<11}\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<20}{precision[i]:<10.5f}{recall[i]:<10.5f}{f1[i]:<10.5f}\n")


def test():
    # Load the model
    model = DatesNet(cfg=cfg)

    # ============== Load the model weights ================
    if not os.path.exists(cfg.TestConfig.checkpoint_name):
        raise FileNotFoundError(f'{cfg.TestConfig.checkpoint_name} file not found')

    checkpoint = torch.load(cfg.TestConfig.checkpoint_name, map_location=cfg.device)
    if 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    else:
        raise KeyError("state_dict is not found in the checkpoint")
    model.load_state_dict(state_dict=state_dict)

    model.eval()
    model.to(cfg.device)

    # ================== Loading the test dataset ==================
    # Use the mean and std computed on the FER+ dataset
    mean = list([0.5015, 0.5015, 0.5015])
    std = list([0.2918, 0.2918, 0.2918])

    data_processor = DataProcessing(cfg)
    test_transform = data_processor.get_test_transform(mean=mean, std=std)
    test_dataset = FERPlusDatasetLoader(cfg=cfg, dataset_name='test', transforms=test_transform)
    test_dataset_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=16, pin_memory=True)
    class_names = test_dataset.get_labels_name()

    # ========== Evaluate the model on the test dataset ============
    with torch.no_grad():
        for (input_data, targets) in tqdm(test_dataset_loader, total=len(test_dataset_loader)):
            if cfg.device == 'cuda':
                model.to(cfg.device)
                input_data = input_data.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # Predicted log probabilities from the model
            predictions = model(input_data)

            # Compute KL-Divergence score
            kl_div = kl_divergence(predictions=predictions, targets=targets)

            # Compute accuracy score, precision, F1, recall scores
            accuracy, precision, recall, f1 = compute_metrics(predictions=predictions, targets=targets)

            # Plot confusion matrix
            plot_confusion_matrix(predictions=predictions, targets=targets, class_names=class_names)

            # Print evaluation metrics
            print("Evaluation Metrics:")
            print(f"KL Divergence: {kl_div:.5f}\n")
            print(f"Overall Accuracy: {accuracy * 100:.5f}%\n\n")
            print(f"{'Class':<20}{'Precision':<11}{'Recall':<11}{'F1-Score':<11}")
            for i, class_name in enumerate(class_names):
                print(f"{class_name:<20}{precision[i]:<10.5f}{recall[i]:<10.5f}{f1[i]:<10.5f}")

            # Dump the evaluation metrics into a txt file
            dump_into_file(kl_div=kl_div,
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1=f1,
                           class_names=class_names,
                           file_path=r'evaluation_metrics.txt')


if __name__ == '__main__':
    test()
