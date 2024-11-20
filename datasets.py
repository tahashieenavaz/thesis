import torch
import numpy as np
import mat73

from functions import path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label - 1, dtype=torch.long)


def build_transforms():
    """
    The `build_transforms` function returns a composition of image transformations including converting
    to PIL image, resizing, converting to tensor, and normalizing.
    :return: A `transforms.Compose` object is being returned, which is a sequence of image
    transformations. The transformations included in the `transforms.Compose` are converting the input
    image to a PIL Image, resizing it to (224, 224) dimensions, converting it to a PyTorch tensor, and
    normalizing the image using the specified mean and standard deviation values.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


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

    return [
        ImageDataset(images=data[0], labels=data[1], transform=build_transforms()),
        len(set(data[1])),
    ]


def dummy_classification_dataset(as_numpy=False):
    x, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=5
    )

    if as_numpy:
        return [x, y]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, y_train, x_test, y_test = (
        torch.tensor(x_train, dtype=torch.float),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(x_test, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    return [train_dataset, test_dataset]
