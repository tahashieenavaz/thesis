import torch
import mat73
import random
from scipy.io import loadmat
import numpy as np

from functions import path
from torchvision import transforms
from torchvision.transforms import v2


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
    def __init__(self, original):
        super(AugmentedDataset, self).__init__()
        self.original = original
        self.pipeline = v2.RandAugment()

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):
        image, label = self.original[idx]
        return self.pipeline(image), label


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


def old_dataset(filename):
    data = loadmat(path(f"../datasets/{filename}"))["DATA"][0]
    images = data[0][0]
    labels = data[1][0]
    labels = list(map(lambda x: x - 1, labels))

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


def planktons(filename="planktons.mat"):
    return old_dataset(filename)


def fibers(filename="fibers.mat"):
    return old_dataset(filename)


def portraits(filename="portraits.mat"):
    data = mat73.loadmat(path(f"../datasets/{filename}"))["DATA"]
    images = data[0]
    labels = data[1]
    labels = list(map(lambda x: x - 1, labels))

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
