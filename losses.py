import torch
from functions import get_device


class MarginEnhancedLogitNormLoss(torch.nn.Module):
    def __init__(self, initial_temperature: float = 0.5, initial_shift: float = 1.0):
        super(MarginEnhancedLogitNormLoss, self).__init__()
        self.device = get_device()
        self.temperature = torch.nn.Parameter(
            torch.tensor(initial_temperature, requires_grad=True)
        )
        self.shift = torch.nn.Parameter(torch.tensor(initial_shift, requires_grad=True))

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + pow(10, -7)
        adjusted_logits = torch.div(x + self.shift, norms) ** (self.temperature)
        return torch.nn.functional.cross_entropy(adjusted_logits, target)
