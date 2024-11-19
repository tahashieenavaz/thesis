import torch


class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(20, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
        )

        self.fc = torch.nn.Linear(32, 5)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)
