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

        return image, torch.tensor(label, dtype=torch.long)


def portraits(filename="portraits.mat"):
    data = mat73.loadmat(path(f"./datasets/{filename}"))["DATA"]
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return ImageDataset(images=data[0], labels=data[1], transform=transform)


def dataset(as_numpy=False):
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
