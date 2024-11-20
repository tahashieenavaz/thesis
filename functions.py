import torch
import json
import os

from classes import Settings


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
