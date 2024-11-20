from sklearn.metrics import roc_auc_score
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
            _, predicted = torch.max(outputs, 1)
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


def get_confusion_matrix(model, dataloader, num_classes: int) -> torch.Tensor:
    """
    The function `get_confusion_matrix` calculates the confusion matrix for a given model and dataloader
    with a specified number of classes.

    :param model: The `model` parameter refers to the neural network model that you have trained for a
    specific task, such as image classification or object detection. This model takes input data and
    produces output predictions
    :param dataloader: A dataloader is an iterable object that allows you to efficiently load and
    iterate over your dataset in batches. It is commonly used in machine learning frameworks like
    PyTorch to feed data to the model during training or evaluation
    :param num_classes: The `num_classes` parameter represents the number of classes in your
    classification problem. It is used to initialize a square matrix of zeros to store the confusion
    matrix, where the rows and columns correspond to the different classes
    :type num_classes: int
    :return: The function `get_confusion_matrix` returns a confusion matrix as a torch.Tensor, which is
    a matrix of shape (num_classes, num_classes) containing the counts of true positive, false positive,
    true negative, and false negative predictions for each class in the classification problem.
    """
    device = get_device()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1)
            for t, p in zip(y_batch.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def get_mcc(model, dataloader, num_classes: int) -> float:
    """
    The function calculates the Matthews Correlation Coefficient (MCC) using a confusion matrix
    generated from a model's predictions on a dataset.

    :param model: The `model` parameter in the `get_mcc` function is typically a machine learning model
    that has been trained on a dataset to perform a specific task, such as image classification or
    natural language processing. It could be a neural network, decision tree, support vector machine, or
    any other type of
    :param dataloader: The `dataloader` parameter in the `get_mcc` function is typically used to pass
    the data loader object that provides batches of data during model evaluation or prediction. It is
    commonly used in machine learning and deep learning workflows to efficiently load and process data
    in batches rather than all at once, which
    :param num_classes: The `get_mcc` function calculates the Matthews Correlation Coefficient (MCC) for
    a given model, dataloader, and number of classes. The MCC is a measure of the quality of binary
    (two-class) classifications, taking into account true positives, true negatives, false positives,
    :return: The function `get_mcc` returns the Matthews Correlation Coefficient (MCC) calculated using
    the confusion matrix generated by the `get_confusion_matrix` function.
    """
    confusion_matrix = get_confusion_matrix(model, dataloader, num_classes)
    tp = confusion_matrix.diag().sum().item()
    total = confusion_matrix.sum().item()
    sum_row = confusion_matrix.sum(dim=0).sum().item()
    sum_col = confusion_matrix.sum(dim=1).sum().item()
    sum_square_row = confusion_matrix.sum(dim=0).pow(2).sum().item()
    sum_square_col = confusion_matrix.sum(dim=1).pow(2).sum().item()
    numerator = tp * total - sum_row * sum_col
    denominator = ((sum_square_row * sum_square_col) ** 0.5) + 1e-10
    return numerator / denominator


def get_roc(model, dataloader) -> float:
    device = get_device()
    y_true = []
    y_scores = []

    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            probabilities = torch.softmax(outputs, dim=1)[
                :, 1
            ]  # Assuming binary classification
            y_scores.extend(probabilities.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    return roc_auc_score(y_true, y_scores)
