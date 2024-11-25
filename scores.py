import torch

from functions import get_device


def energy_score(logits):
    """
    The function `energy_score` calculates the energy score based on the logits provided.

    :param logits: Logits typically refer to the raw, unnormalized predictions generated by a model
    before applying a softmax function. In the provided code snippet, the `energy_score` function takes
    logits as input and calculates the energy score based on the softmax probabilities derived from the
    logits
    :return: The function `energy_score` returns the negative natural logarithm of the maximum
    probability value in the input logits after applying softmax along the last dimension.
    """
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return -1 * torch.log(torch.max(probabilities, dim=-1).values)


def get_scores(model, score_fn: callable, dataloader) -> torch.Tensor:
    """
    The function `get_scores` takes a model, a scoring function, and a dataloader, and returns the
    scores obtained by applying the scoring function to the model's predictions on the data provided by
    the dataloader.

    :param model: A PyTorch model that you want to evaluate
    :param score_fn: The `score_fn` parameter is a callable function that takes the logits output by the
    model and computes scores based on those logits. This function could perform tasks such as
    calculating probabilities, applying a threshold, or any other operation to derive scores from the
    model's output
    :type score_fn: callable
    :param dataloader: A dataloader is an object in PyTorch that helps in efficiently loading and
    iterating over a dataset during training or evaluation. It typically provides functionalities like
    shuffling, batching, and parallel data loading. You can create a dataloader in PyTorch using the
    `torch.utils.data.DataLoader
    :return: The function `get_scores` returns a torch.Tensor containing the scores calculated by the
    `score_fn` for each batch of data processed by the model.
    """
    device = get_device()
    scores = torch.tensor([], device=device)
    model.eval()
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            batch_scores = score_fn(logits)
            scores = torch.cat((batch_scores.flatten(), scores.flatten()))
    return scores
