import torch
import numpy as np
import mat73
import random

from functions import path
from torchvision import transforms
from torchvision.datasets import CIFAR10


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augmentations=5):
        super(AugmentedDataset, self).__init__()
        self.original_dataset = original_dataset
        self.num_augmentations = num_augmentations
        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                ),
                transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
            ]
        )

        self.augmented_data = []
        self.augmented_labels = []
        self._create_augmented_dataset()

    def _create_augmented_dataset(self):
        for i in range(len(self.original_dataset)):
            image, label = self.original_dataset[i]
            for _ in range(self.num_augmentations):
                augmented_image = self.augmentations(image)
                self.augmented_data.append(augmented_image)
                self.augmented_labels.append(label)

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        return self.augmented_data[idx], self.augmented_labels[idx]


def build_transforms(convert_to_image: bool = True):
    """
    The `build_transforms` function returns a composition of image transformations including converting
    to PIL image, resizing, converting to tensor, and normalizing.
    :return: A `transforms.Compose` object is being returned, which is a sequence of image
    transformations. The transformations included in the `transforms.Compose` are converting the input
    image to a PIL Image, resizing it to (224, 224) dimensions, converting it to a PyTorch tensor, and
    normalizing the image using the specified mean and standard deviation values.
    """
    transforms_list = []

    if convert_to_image:
        transforms_list.append(transforms.ToPILImage())

    transforms_list += [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(transforms_list)


def portraits(filename="portraits.mat"):
    """
    The function `portraits` loads image data from a .mat file and returns an ImageDataset object along
    with the number of unique labels in the data.

    :param filename: The `filename` parameter is a string that represents the name of the file
    containing the portraits data. The default value for this parameter is "portraits.mat", defaults to
    portraits.mat (optional)
    :return: The function `portraits` is returning a list containing two elements:
    1. An `ImageDataset` object created with images and labels extracted from the loaded data, along
    with a specified transformation.
    2. The number of unique labels present in the data.
    """
    data = mat73.loadmat(path(f"./datasets/{filename}"))["DATA"]
    images = data[0]
    labels = list(map(lambda x: x - 1, data[1]))

    indices = list(range(len(labels)))
    random.shuffle(indices)

    images_shuffled = [images[i] for i in indices]
    labels_shuffled = [labels[i] for i in indices]

    return [
        ImageDataset(
            images=images_shuffled, labels=labels_shuffled, transform=build_transforms()
        ),
        len(set(labels)),
    ]


def cifar10():
    train = CIFAR10(
        root="./data", train=True, download=True, transform=build_transforms(False)
    )
    test = CIFAR10(
        root="./data", train=False, download=True, transform=build_transforms(False)
    )
    dataset = torch.utils.data.ConcatDataset([train, test])
    return [dataset, 10]
