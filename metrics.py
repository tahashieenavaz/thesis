from functions import get_device
import torch


def get_accuracy(model, dataloader) -> float:
    """
    This function calculates the accuracy of a given model on a test dataset.

    :param model: A machine learning model that you have trained for a specific task, such as image
    classification or sentiment analysis
    :param test: The `test` parameter in the `get_accuracy` function is typically a dataset containing
    test samples and their corresponding labels. It is used to evaluate the accuracy of the `model` on
    this test dataset
    :return: The function `get_accuracy` returns the accuracy of a given model on a test dataset,
    calculated as the percentage of correct predictions.
    """
    device = get_device()
    total = 0
    correct = 0
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total * 100


def get_loss(model, dataloader, loss_fn) -> float:
    """
    This function calculates the cumulative loss of a given model on a test dataset.

    :param model: A machine learning model that you have trained for a specific task
    :param test: The test dataset, containing samples and their corresponding labels
    :param loss_fn: The loss function used to evaluate the model's predictions against the ground truth
    :return: The cumulative loss of the model on the test dataset
    """
    device = get_device()
    cumulative_loss = 0.0
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            cumulative_loss += loss.item()
    return cumulative_loss
