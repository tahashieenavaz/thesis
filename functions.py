import torch
import json
import os


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_model(model):
    for layer in model.layers:
        layer.reset_parameters()


def current_dir() -> str:
    return os.getcwd()


def path(relative: str) -> str:
    return os.path.join(current_dir(), relative)


def tensor(data, dtype=None):
    if not dtype:
        if "numpy" in str(type(data)):
            return torch.from_numpy(data)

        return torch.tensor(data)

    return torch.tensor(data, dtype=dtype)


def load_settings(**kwargs):
    class Settings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    with open(path("./settings.json"), "r") as f:
        initial_settings = json.load(f)

    for key, value in kwargs:
        initial_settings[key] = value

    return Settings(**initial_settings)
