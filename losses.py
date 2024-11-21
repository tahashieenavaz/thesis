import torch
from functions import get_device


class LogitNormLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0, margin: float = 0.1):
        super(LogitNormLoss, self).__init__()
        self.device = get_device()
        self.temperature = torch.nn.Parameter(
            torch.tensor(temperature, requires_grad=True)
        )
        self.margin = torch.nn.Parameter(torch.tensor(margin, requires_grad=True))

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + pow(10, -7)
        logit_norm = torch.div(x - self.margin, norms) / self.temperature
        return torch.nn.functional.cross_entropy(logit_norm, target)
