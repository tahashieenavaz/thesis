import torch
from functions import get_device


class LogitNormLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(LogitNormLoss, self).__init__()
        self.device = get_device()
        self.temperature = temperature

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + pow(10, -7)
        logit_norm = torch.div(x, norms) / self.temperature
        return torch.nn.functional.cross_entropy(logit_norm, target)
