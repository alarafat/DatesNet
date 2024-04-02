import cv2
import warnings

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from data.dataset_loader import FERPlusDatasetLoader


class DataProcessing(object):
    """
    Data pre-processing and augmentation of the input image.

    Methods:
        compute_db_mean_std(): Computes the global mean and standard deviation across all images in the
            training dataset for each channel.

        get_online_aug_transform(mean=None, std=None): Returns a torchvision.transforms.Compose object
            for online data augmentation during training, including random horizontal flip, random resized
            crop, random rotation, random adjust sharpness, color jitter, and normalization.

        get_valid_transform(mean=None, std=None): Returns a torchvision.transforms.Compose object for
            validation data augmentation, including histogram equalization, grayscale to RGB conversion,
            tensor conversion, and normalization.

        get_test_transform(mean=None, std=None): Returns a torchvision.transforms.Compose object for
            test data augmentation, identical to the validation augmentation.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.mean = torch.tensor([0, 0, 0])
        self.std = torch.tensor([1, 1, 1])
        self.preprocess_transform = v2.Compose([
            self.HistogramEqualization(),
            self.Gray2RGB(),
            v2.ToTensor(),
            v2.Normalize(mean=self.mean, std=self.std),
        ])
        self.is_global_mean_std_calc = False

    def compute_db_mean_std(self):
        """
            Computes the global mean and standard deviation of the dataset used in training.

            This method calculates the mean and standard deviation across all images in the training dataset
            for each channel (assuming RGB images).
        """

        dataset = FERPlusDatasetLoader(cfg=self.cfg, dataset_name='train', transforms=self.preprocess_transform)
        dataset_loader = DataLoader(dataset,
                                    batch_size=self.cfg.TrainConfig.batch_size,
                                    shuffle=False,
                                    num_workers=self.cfg.TrainConfig.workers,
                                    pin_memory=True)

        summ = torch.tensor([0.0, 0.0, 0.0])
        sum_square = torch.tensor([0.0, 0.0, 0.0])

        for data, _ in dataset_loader:
            summ += torch.mean(data, dim=[0, 2, 3])
            sum_square += torch.mean(data ** 2, dim=[0, 2, 3])
        self.mean = summ / len(dataset_loader)
        self.std = torch.sqrt((sum_square / len(dataset_loader)) - self.mean ** 2)
        self.is_global_mean_std_calc = True
        return self.mean, self.std

    class HistogramEqualization(object):
        def __call__(self, in_image):
            """
            Apply histogram equalization to the input image
            """
            eq_image = cv2.equalizeHist(in_image)
            out_image = cv2.normalize(eq_image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # clahe_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # eq_image = clahe_image.apply(in_image)
            # out_image = cv2.normalize(eq_image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            return torch.from_numpy(out_image).float().unsqueeze(0)

    class Gray2RGB(object):
        """"
        Converts 1-channel image to 3-channels image
        """

        def __call__(self, in_image):
            return in_image.expand(3, -1, -1)

    def get_online_aug_transform(self, mean=None, std=None):
        """"
        Returns the online augmentation transforms for the training
        """
        if not self.is_global_mean_std_calc:
            warnings.warn('You are not using db global mean and std, rather using 0 mean with unit gaussian. '
                          'Use compute_db_mean_std() before calling the get_online_aug_transform() method')
        mean = mean if mean is not None else self.mean
        std = std if std is not None else self.std
        augmentation_transform = v2.Compose([
            self.HistogramEqualization(),  # Histogram equalization
            self.Gray2RGB(),  # Convert to a 3-channels image
            v2.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with probability of 0.5
            v2.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomRotation(degrees=15),  # Rotates the image by up to 15 degrees
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # Randomly adjusts sharpness
            v2.ColorJitter(brightness=0.2, contrast=0.2),  # Randomly changes brightness and contrast
            # v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std),  # Normalize tensors
        ])

        return augmentation_transform

    def get_valid_transform(self, mean=None, std=None):
        """
        Returns the augmentation transforms for validation
        """
        mean = mean if mean is not None else self.mean
        std = std if std is not None else self.std
        augmentation_transform = v2.Compose([
            self.HistogramEqualization(),
            self.Gray2RGB(),
            v2.ToTensor(),  # Convert images to PyTorch tensors
            v2.Normalize(mean=mean, std=std),  # Normalize tensors
        ])
        return augmentation_transform

    def get_test_transform(self, mean=None, std=None):
        """
        Returns the augmentation transforms for test. This is exactly same as the validation augmentation
        """
        return self.get_valid_transform(mean=mean, std=std)
