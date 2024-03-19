"""
Data loader module for Celeba
"""
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


class celeba_dataset(Dataset):
    def __init__(
        self,
        is_text,
        transform=None,
        num_samples=5000,
        configs="",
        training=True,
        test_size=None,
        alpha=1.0,
        remove_node=None,
        seed=42,
    ):
        random.seed(seed)
        self.is_text = is_text
        self.text_map = [
            {"0": "not male", "1": "male"},
            {"0": "smiling", "1": "not smiling"},
            {"0": "black hair", "1": "blond hair"},
        ]

        self.training = training
        self.test_size = test_size
        prefix = "celeba"
        ext = ".jpg"
        self.dataset = "celeba-3classes-10000"

        if self.training:
            self.train_image_paths = []
            for config in configs:
                if config == "000" and alpha != 1500 and remove_node != "100":
                    path_pattern = f"input/{self.dataset}/train_{remove_node}/{prefix}_000_*{ext}"
                    new_paths = glob.glob(path_pattern)
                    if remove_node == config:
                        new_paths = new_paths[:alpha]
                else:
                    path_pattern = (
                        f"input/{self.dataset}/train/{prefix}_{config}_*{ext}"
                    )
                    new_paths = glob.glob(path_pattern)
                    if remove_node == config:
                        new_paths = new_paths[:alpha]
                self.train_image_paths.extend(new_paths)
            self.len_data = len(self.train_image_paths)

        else:
            self.test_image_paths = glob.glob(
                f"input/{self.dataset}/test/{prefix}_{configs}_*{ext}"
            )
            self.len_data = len(self.test_image_paths)

        self.num_samples = num_samples
        self.transform = transform

    def __getitem__(self, index):
        if self.training:
            ipath = random.randint(0, len(self.train_image_paths) - 1)
            img_path = self.train_image_paths[ipath]

        else:
            ipath = random.randint(0, len(self.test_image_paths) - 1)
            img_path = self.test_image_paths[ipath]

        try:
            img = Image.open(img_path)  # .convert('RGB')
        except:
            print(img_path)
            sys.exit()

        if self.transform is not None:
            img = self.transform(img)

        name_labels = img_path.split("_")[-2]
        label = {i: int(name_labels[i]) for i in range(3)}
        if self.is_text:
            label = ", ".join(
                [self.text_map[i][name_labels[i]] for i in range(3)]
            )

        return img, label

    def __len__(self):
        return self.num_samples
