import torch
from functions import get_device


class MarginEnhancedLogitNormLoss(torch.nn.Module):
    def __init__(
        self,
        initial_temperature: float = 1.0,
        initial_margin: float = 0.1,
        initial_factor: float = 1.0,
        initial_offset: float = 0.0,
    ):
        super(MarginEnhancedLogitNormLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = torch.nn.Parameter(
            torch.tensor(initial_temperature, requires_grad=True)
        )
        self.margin = torch.nn.Parameter(
            torch.tensor(initial_margin, requires_grad=True)
        )
        self.factor = torch.nn.Parameter(
            torch.tensor(initial_factor, requires_grad=True)
        )
        self.offset = torch.nn.Parameter(
            torch.tensor(initial_offset, requires_grad=True)
        )

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        adjusted_logits = (x - self.factor * self.margin + self.offset) / norms
        logit_norm = adjusted_logits / self.temperature
        return torch.nn.functional.cross_entropy(logit_norm, target)
