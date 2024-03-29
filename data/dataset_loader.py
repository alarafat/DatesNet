import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data as data


class FERPlusDatasetLoader(data.Dataset):
    """
    A custom PyTorch dataset loader for the FER+ dataset, supporting data loading and preprocessing for
    training, validation, and testing purposes.
    """

    def __init__(self, cfg, dataset_name: str, transforms=None):
        assert dataset_name in ['train', 'test', 'val'], "Dataset name can be in [train/test/val]"
        self.cfg = cfg
        dataset_name_pairs = dict(train='Training',
                                  test='PublicTest',
                                  val='PrivateTest')
        self.image_shape = cfg.DatasetConfig.image_shape
        self.dataset_name = dataset_name_pairs[dataset_name]
        self.fer = pd.read_csv(cfg.DatasetConfig.dataset_dir + r'\fer2013.csv')
        self.fer_plus = pd.read_csv(cfg.DatasetConfig.dataset_dir + r'\fer2013new.csv')
        self.classes_name = self.fer_plus.columns[2: 2 + cfg.ModelConfig.n_classes].tolist()
        self.transforms = transforms

        self.data, self.labels = self.load_data()

    def load_data(self) -> (np.ndarray, np.ndarray):
        """
        Loads the data from the FER2013 and FER2013new CSV files, aligns the enhanced labels with the images,
        and preprocesses the data according to the specified image shape and label encoding.

        Returns:
            tuple: A tuple containing two numpy arrays: the preprocessed images and their corresponding encoded labels.
        """
        images = list()
        labels = list()

        curr_fer_plus = self.fer_plus[self.fer_plus["Usage"] == self.dataset_name]
        for idx, (image_name, image_data) in tqdm(enumerate(zip(curr_fer_plus['Image name'], self.fer['pixels'])), total=len(curr_fer_plus)):
            if pd.isna(image_name) or self.fer_plus.iloc[idx][2: 2 + self.cfg.ModelConfig.n_classes].sum() == 0:
                continue

            encoded_label = self.encode_labels(np.array([self.fer_plus.loc[idx, each_class] for each_class in self.classes_name]))
            if encoded_label.size:
                images.append(np.fromstring(image_data, dtype=np.uint8, sep=' ').reshape(self.image_shape))
                labels.append(encoded_label)

        assert len(images) == len(labels), "Number of loaded images does not match the number of labels. Something went wrong!"
        return np.array(images), np.array(labels)

    @staticmethod
    def encode_labels(labels: np.ndarray) -> np.ndarray:
        """
        Encodes the labels into a normalized probability distribution, where each label's probability is proportional to its vote count.

        Parameters:
            labels (np.ndarray): An array containing vote counts for each class for a single image.

        Returns:
            np.ndarray: A normalized array representing the probability distribution of the labels.
        """
        labels[labels <= 1] = 0
        if not sum(labels):
            return np.array([])

        labels = labels / sum(labels)

        assert np.isclose(sum(labels), 1), "Labels dont make a true prob distribution"

        return labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of images in the dataset subset.
        """
        return len(self.data)

    def get_labels_name(self):
        """
        Returns the names of the classes (emotions) in the dataset.

        Returns:
            list: A list containing the names of the classes in the dataset.
        """
        return self.classes_name

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves the image and its label at the specified index, applying any specified transformations to the image.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        image = self.data[index]
        # image = np.repeat(np.expand_dims(self.data[index], -1), 3, axis=-1)
        label = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image.to(), label
