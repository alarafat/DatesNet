import os
import numpy as np
import cv2
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from lib.datesnet import DatesNet
# from lib.datesnet_mlp import DatesNet
# from lib.datesnet_vgg import DatesNet

from config.datesnet_config import Config as cfg
from data.data_processing import DataProcessing
from data.dataset_loader import FERPlusDatasetLoader
from utils.evaluation_metrics import kl_divergence, compute_accuracy, compute_metrics, compute_confusion_matrix


def plot_confusion_matrix(confusion_mat, class_names):
    """
    Plots a confusion matrix using seaborn's heatmap.

    This function saves the plotted confusion matrix to a file named 'confusion_matrix.png'
    in the output directory specified in the global configuration (`cfg.output_dir`),
    and also displays the plot.
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_mat.astype(float), annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(cfg.output_dir, r'confusion_matrix.png'))
    plt.show(block=True)


def dump_into_file(kl_div, accuracy, precision, recall, f1, class_names, file_path=r'evaluation_metrics.txt'):
    """
    Writes the evaluation metrics into a file.

    Parameters:
        kl_div (float): The KL Divergence score.
        accuracy (float): The overall accuracy score.
        precision (list of float): Precision scores for each class.
        recall (list of float): Recall scores for each class.
        f1 (list of float): F1 scores for each class.
        class_names (list of str): Names of the classes.
        file_path (str, optional): Path to the file where metrics will be written. Defaults to 'evaluation_metrics.txt'.
    """
    with open(os.path.join(cfg.output_dir, file_path), 'w') as f:
        f.write("Evaluation Metrics:\n")
        print(f"KL Divergence: {kl_div:.5f}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.5f}%\n\n")
        f.write(f"{'Class':<20}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}\n")
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

    if cfg.TestConfig.use_single_image:
        assert len(cfg.TestConfig.class_names) == 0, "You need to se the class names if you want to use a single image"
        class_names = cfg.TestConfig.class_names
        gray_image = cv2.imread(cfg.TestConfig.image_file_name, cv2.IMREAD_GRAYSCALE)

        gray_image = np.expand_dims(gray_image, axis=0)  # Add batch dimension
        rgb_image = np.expand_dims(gray_image, axis=0)  # Add channel dimension
        in_image = torch.tensor(rgb_image, dtype=torch.float32)

        # Normalize the image
        transform = v2.Normalize(mean=mean, std=std)
        in_image = transform(in_image)

        # Move the image to the device and evaluate
        in_image = in_image.to(cfg.device)
        with torch.no_grad():
            predictions = model(in_image)
    else:
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
    accuracy = compute_accuracy(predictions=predictions, targets=targets)
    precision, recall, f1 = compute_metrics(predictions=predictions, targets=targets)

    # Compute confusion matrix
    conf_mat = compute_confusion_matrix(predictions=predictions, targets=targets)
    plot_confusion_matrix(confusion_mat=conf_mat, class_names=class_names)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(f"KL Divergence: {kl_div:.5f}\n")
    print(f"Overall Accuracy: {accuracy * 100:.5f}%\n\n")
    print(f"{'Class':<20}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
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
