import os
import json
import random
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from cg_constants import ROOT_DIR


class shapes_dataset(Dataset):
    def __init__(
        self,
        text,
        transform=None,
        num_samples=5000,
        configs="",
        training=True,
        test_size=None,
        alpha=1.0,
        remove_node=None,
    ):
        dataset = "single-body_2d_3classes"
        self.text = text
        if self.text:
            raise RuntimeError("Unsupported for shapes.")

        self.training = training
        self.test_size = test_size

        prefix = "CLEVR"
        ext = ".png"
        if training:
            self.train_image_paths = []
            for config in configs:
                if config == "000" and alpha != 1500 and remove_node != "100":
                    path_pattern = f"input/{dataset}/train_{remove_node}/{prefix}_000_*{ext}"
                    new_paths = glob.glob(path_pattern)
                    if remove_node == config:
                        new_paths = new_paths[:alpha]
                else:
                    path_pattern = (
                        f"input/{dataset}/train/{prefix}_{config}_*{ext}"
                    )
                    new_paths = glob.glob(os.path.join(ROOT_DIR, path_pattern))
                    if remove_node == config:
                        new_paths = new_paths[:alpha]
                self.train_image_paths.extend(new_paths)
                print(len(self.train_image_paths))
            self.len_data = len(self.train_image_paths)

        else:
            self.test_image_paths = glob.glob(
                os.path.join(
                    ROOT_DIR, f"input/{dataset}/test/{prefix}_{configs}_*{ext}"
                )
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

        img = Image.open(img_path)  # .convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        name_labels = img_path.split("_")[-2]

        with open(img_path.replace(".png", ".json"), "r") as f:
            my_dict = json.loads(f.read())
            _size = my_dict[0]
            _color = my_dict[1][:3]

        if self.training:
            size, color = _size, _color
        else:
            # Define colors mapping
            colors_map = {
                "0": [0.9, 0.1, 0.1],
                "1": [0.1, 0.1, 0.9],
                "2": [0.1, 0.9, 0.1],
            }
            # Assign size and color based on label values
            size = 2.6 if int(name_labels[2]) == 0 else self.test_size
            color = colors_map[name_labels[1]]

        # Convert size and color to numpy arrays
        size = np.array(size, dtype=np.float32)
        color = np.array(color, dtype=np.float32)

        # Create the label dictionary
        label = {0: int(name_labels[0]), 1: color, 2: size}
        return img, label

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = my_dataset(
        transform,
        dataset="single-body_2d_3classes",
        configs=["000", "010", "100", "001"],
        remove_node="010",
    )
    dataloader = DataLoader(dataset, batch_size=4)

    for img, label in dataloader:
        print("label=", label)
        print(img.shape)
        plt.imshow(np.transpose(img[0].numpy(), (2, 1, 0)))
        plt.show()
        print("img.shape=", img.shape)
        exit()