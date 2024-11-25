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


class MarginTemperatureEnhancedHingeLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0, tradeoff: float = 0.1):
        super(MarginTemperatureEnhancedHingeLoss, self).__init__()
        self.margin = margin
        self.tradeoff = tradeoff
        self.phi = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.theta = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, logits, targets):
        logits = logits + torch.relu(self.theta)
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + pow(10, -7)
        logits = self.phi * torch.div(logits, norms)

        batch_size = logits.size()[0]
        true_class_logits = logits[torch.arange(batch_size), targets].unsqueeze(1)

        margins = self.margin + logits - true_class_logits
        margins[torch.arange(batch_size), targets] = 0

        return (
            torch.clamp(margins, min=0).sum(dim=1).mean() - self.tradeoff * logits.std()
        )
