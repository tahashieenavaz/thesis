import torch
from functions import get_device


class MarginEnhancedLogitNormLoss(torch.nn.Module):
    def __init__(self, initial_temperature: float = 1.0):
        super(MarginEnhancedLogitNormLoss, self).__init__()
        self.device = get_device()
        self.temperature = torch.nn.Parameter(
            torch.tensor(initial_temperature, requires_grad=True)
        )
        self.booster = torch.nn.Parameter(torch.tensor(2.0, requires_grad=True))

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + pow(10, -7)
        adjusted_logits = torch.div(x, self.booster * norms) * (1 + self.temperature)
        return torch.nn.functional.cross_entropy(adjusted_logits, target)
