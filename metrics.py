import numpy as np
import torch

from functions import get_device
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score


def get_accuracy(model, dataloader) -> float:
    model.eval()
    device = get_device()
    total = 0
    correct = 0
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)

            print("Model output shape:", outputs.shape)  # Debug output
            _, predicted = torch.max(outputs, 1)
            print("Predicted:", predicted[:10])
            print("True labels:", y_batch[:10])

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            print(
                f"Batch accuracy: {(predicted == y_batch).sum().item() / y_batch.size(0) * 100:.2f}%"
            )

    print(f"Total: {total}, Correct: {correct}")
    return correct / total * 100


# def get_accuracy(model, dataloader) -> float:
#     """
#     This function calculates the accuracy of a given model on a test dataset.

#     :param model: A machine learning model that you have trained for a specific task, such as image
#     classification or sentiment analysis
#     :param test: The `test` parameter in the `get_accuracy` function is typically a dataset containing
#     test samples and their corresponding labels. It is used to evaluate the accuracy of the `model` on
#     this test dataset
#     :return: The function `get_accuracy` returns the accuracy of a given model on a test dataset,
#     calculated as the percentage of correct predictions.
#     """
#     device = get_device()
#     total = 0
#     correct = 0
#     with torch.inference_mode():
#         for x_batch, y_batch in dataloader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             outputs = model(x_batch)
#             _, predicted = torch.max(outputs, 1)
#             total += y_batch.size(0)
#             correct += (predicted == y_batch).sum().item()
#     return correct / total * 100


def get_loss(model, dataloader, loss_fn) -> float:
    """
    This function calculates the cumulative loss of a given model on a test dataset.

    :param model: A machine learning model that you have trained for a specific task
    :param test: The test dataset, containing samples and their corresponding labels
    :param loss_fn: The loss function used to evaluate the model's predictions against the ground truth
    :return: The cumulative loss of the model on the test dataset
    """
    model.eval()
    device = get_device()
    cumulative_loss = 0.0
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            cumulative_loss += loss.item()
    return cumulative_loss


def get_kappa(model, dataloader):
    """
    Computes Cohen's Kappa for a deep learning model.

    :param model: Trained PyTorch model
    :param dataloader: DataLoader object for the test dataset
    :return: Cohen's Kappa score
    """
    model.eval()
    device = get_device()
    all_true = []
    all_pred = []

    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1)

            all_true.extend(y_batch.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    kappa_score = cohen_kappa_score(all_true, all_pred)
    return kappa_score


def get_f1(model, dataloader, average="weighted"):
    """
    Computes the F1 Score for a deep learning model.

    :param model: Trained PyTorch model
    :param dataloader: DataLoader object for the test dataset
    :param average: Averaging method for F1 score ('micro', 'macro', 'weighted', or None)
    :return: F1 Score
    """
    model.eval()
    device = get_device()
    all_true = []
    all_pred = []

    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1)

            all_true.extend(y_batch.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    f1 = f1_score(all_true, all_pred, average=average)
    return f1
