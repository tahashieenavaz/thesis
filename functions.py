import torch
import json
import numpy as np
import hashlib
import os

from torchvision.models import resnet50 as resnet
from classes import Settings


def sha1(text: str):
    """
    The function `sha1` takes a string input, encodes it using SHA-1 algorithm, and returns the
    hexadecimal digest.

    :param text: The `sha1` function you provided takes a string `text` as input and calculates the
    SHA-1 hash of that text. The `text` parameter is the string for which you want to calculate the
    SHA-1 hash
    :type text: str
    :return: The `sha1` function is returning the SHA-1 hash digest of the input text as a hexadecimal
    string.
    """
    return hashlib.sha1(text.encode()).hexdigest()


def seed():
    """
    The function `seed()` generates a seed value based on a given string and uses it to set the random
    seed for PyTorch and NumPy.
    """
    base = "Taha Shieenavaz and Loris Nanni"
    seed_value = int("".join(list(filter(lambda x: str.isdigit(x), sha1(base)))[:8]))
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)


def create_folder(folder_path: str) -> None:
    """
    The function `create_folder` creates a new folder at the specified path if it does not already
    exist.

    :param folder_path: The `folder_path` parameter in the `create_folder` function is a string that
    represents the path of the folder that you want to create. This function checks if the folder
    already exists, and if it doesn't, it creates the folder at the specified path
    :type folder_path: str
    """
    if not os.path.exists(path(folder_path)):
        os.makedirs(folder_path)


def get_device():
    """
    The function `get_device()` returns the appropriate device (CUDA or CPU) based on the availability
    of CUDA in PyTorch.
    :return: The `get_device` function returns a torch device, which is either "cuda" if a CUDA-enabled
    GPU is available, or "cpu" if not.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flush(message: str) -> None:
    """
    The `flush` function in Python prints a message with the `flush` argument set to `True`.

    :param message: The `flush` function takes a single parameter `message` of type `str`. This function
    prints the message to the standard output stream with the `flush` parameter set to `True`, which
    forces the output to be flushed immediately
    :type message: str
    """
    print(message, flush=True)


def reset_model(model) -> None:
    """
    The function `reset_model` iterates through the layers of a model and resets their parameters.

    :param model: The `reset_model` function you provided seems to be intended to reset the parameters
    of each layer in a given model. However, it seems like the implementation is incomplete as the
    `reset_parameters()` method is not a standard method in most deep learning frameworks like
    TensorFlow or PyTorch
    """
    for layer in model.layers:
        layer.reset_parameters()


def current_dir() -> str:
    """
    The function `current_dir` returns the current working directory in Python.
    :return: The function `current_dir()` is returning the current working directory using the
    `os.getcwd()` function.
    """
    return os.getcwd()


def path(relative: str) -> str:
    """
    The function `path` takes a relative path as input and returns the absolute path by joining it with
    the current directory.

    :param relative: The `relative` parameter in the `path` function is a string that represents the
    relative path to a file or directory
    :type relative: str
    :return: The function `path` is returning a string that represents the path formed by joining the
    current directory path with the relative path provided as an argument.
    """
    return os.path.join(current_dir(), relative)


def tensor(data, dtype=None) -> torch.tensor:
    """
    The function `tensor` converts input data into a PyTorch tensor, optionally specifying the data
    type.

    :param data: The `data` parameter in the `tensor` function is the input data that you want to
    convert into a PyTorch tensor. It can be a NumPy array, a Python list, or any other iterable data
    structure that can be converted into a tensor
    :param dtype: The `dtype` parameter in the `tensor` function is used to specify the data type of the
    tensor that will be created. If `dtype` is provided, the function will create a tensor with the
    specified data type. If `dtype` is not provided (i.e., it is `None
    :return: The function `tensor` returns a torch tensor object. If the `dtype` parameter is not
    provided, it checks if the input data is a numpy array and converts it to a torch tensor using
    `torch.from_numpy(data)` if it is. Otherwise, it creates a torch tensor from the input data using
    `torch.tensor(data)`. If the `dtype` parameter is provided, it creates a torch
    """
    if not dtype:
        if "numpy" in str(type(data)):
            return torch.from_numpy(data)

        return torch.tensor(data)

    return torch.tensor(data, dtype=dtype)


def load_settings(**kwargs) -> Settings:
    """
    The function `load_settings` loads settings from a JSON file, updates them with any keyword
    arguments provided, and returns a `Settings` object.
    :return: An instance of the Settings class with the updated settings based on the provided keyword
    arguments.
    """
    with open(path("./settings.json"), "r") as f:
        initial_settings = json.load(f)

    for key, value in kwargs:
        initial_settings[key] = value

    return Settings(**initial_settings)


def build_model(num_classes: int):
    """
    The function `build_model` creates a ResNet model with a specified number of output classes.

    :param num_classes: The `num_classes` parameter in the `build_model` function represents the number
    of classes in the classification task for which you want to build a model. This parameter is used to
    modify the last fully connected layer of the ResNet model to output predictions for the specified
    number of classes
    :type num_classes: int
    :return: The function `build_model` returns a ResNet model with the specified number of output
    classes.
    """
    model = resnet(weights="IMAGENET1K_V1").to(get_device())
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(get_device())
    return model


def build_optimizer(model, criterion, lr: float):
    cnn_params = [
        param for name, param in model.named_parameters() if not name.startswith("fc")
    ]
    fc_params = [
        param for name, param in model.named_parameters() if name.startswith("fc")
    ]
    return torch.optim.Adam(
        [
            {"params": cnn_params, "lr": lr},
            {"params": fc_params, "lr": lr * 20},
            {"params": [criterion.temperature, criterion.margin], "lr": lr},
        ],
    )
