import torch

from functions import get_device


def energy_score(logits):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return -1 * torch.log(torch.max(probabilities, dim=-1).values)


def get_scores(model, score_fn: callable, dataloader) -> torch.Tensor:
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
