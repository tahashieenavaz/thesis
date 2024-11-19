import torch


def energy_score(logits):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return -1 * torch.log(torch.max(probabilities, dim=-1).values)
