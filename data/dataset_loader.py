import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data


class FERPlusDatasetLoader(data.Dataset):
    def __init__(self, cfg, dataset_name: str, transforms=None):
        assert dataset_name in ['train', 'test', 'val'], "Dataset name can be in [train/test/val]"
        self.cfg = cfg
        dataset_name_pairs = dict(train='Training',
                                  test='PublicTest',
                                  val='PrivateTest')
        self.image_shape = cfg.DatasetConfig.image_shape
        # self.total_votes = cfg.DatasetConfig.votes_count

        self.dataset_name = dataset_name_pairs[dataset_name]
        self.fer = pd.read_csv(cfg.DatasetConfig.dataset_dir + r'\fer2013.csv')
        self.fer_plus = pd.read_csv(cfg.DatasetConfig.dataset_dir + r'\fer2013new.csv')
        self.classes_name = self.fer_plus.columns[2: 2 + cfg.ModelConfig.n_classes].tolist()
        self.transforms = transforms

        self.data, self.labels = self.load_data()

    def load_data(self) -> (np.ndarray, np.ndarray):
        images = list()
        labels = list()

        curr_fer_plus = self.fer_plus[self.fer_plus["Usage"] == self.dataset_name]
        for idx, (image_name, image_data) in tqdm(enumerate(zip(curr_fer_plus['Image name'], self.fer['pixels'])), total=len(curr_fer_plus)):
            if pd.isna(image_name) or self.fer_plus.iloc[idx][2: 2 + self.cfg.ModelConfig.n_classes].sum() == 0:
                continue
            images.append(np.fromstring(image_data, dtype=np.uint8, sep=' ').reshape(self.image_shape))
            labels.append(self.encode_labels(np.array([self.fer_plus.loc[idx, each_class] for each_class in self.classes_name])))

        assert len(images) == len(labels), "Number of loaded images does not match the number of labels. Something went wrong!"
        return np.array(images), np.array(labels)

    @staticmethod
    def encode_labels(labels: np.ndarray) -> np.ndarray:
        labels[labels <= 1] = 0
        labels = labels / sum(labels)

        assert np.isclose(sum(labels), 1), "Labels dont make a true prob distribution"

        return labels

    def __len__(self):
        return len(self.data)

    def get_labels_name(self):
        return self.classes_name

    def __getitem__(self, index: int) -> tuple:
        image = self.data[index]
        # image = np.repeat(np.expand_dims(self.data[index], -1), 3, axis=-1)
        label = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image.to(), label
