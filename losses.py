import torch
from functions import get_device


class MarginEnhancedLogitNormLoss(torch.nn.Module):
    def __init__(self, initial_temperature: float = 1.0, initial_margin: float = 0.1):
        super(MarginEnhancedLogitNormLoss, self).__init__()
        self.device = get_device()
        self.temperature = torch.nn.Parameter(
            torch.tensor(initial_temperature, requires_grad=True)
        )
        self.margin = torch.nn.Parameter(
            torch.tensor(initial_margin, requires_grad=True)
        )

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + pow(10, -7)
        logit_norm = torch.div(x - self.margin, norms) / self.temperature
        return torch.nn.functional.cross_entropy(logit_norm, target)
