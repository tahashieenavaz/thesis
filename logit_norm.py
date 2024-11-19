import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from datasets import portraits
from functions import get_device
from functions import load_settings
from metrics import get_accuracy
from scores import energy_score
from torchvision.models import resnet18 as resnet
from losses import LogitNormLoss
from sklearn.model_selection import KFold

torch.manual_seed(2092671)
np.random.seed(2092671)

dataset = portraits()
device = get_device()
settings = load_settings()

kf = KFold(n_splits=settings.folds)


energies = []
accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, test_idx)

    model = resnet().to(device)

    criterion = LogitNormLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [criterion.temperature],
        settings.lr,
        weight_decay=pow(10, -5),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)

    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=settings.batch_size
    )

    for epoch in tqdm(range(settings.epochs)):
        train_dataloader = torch.utils.data.DataLoader(
            train, shuffle=True, batch_size=settings.batch_size
        )

        model.train()
        for x_batch, y_batch in tqdm(train_dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # epochs end
    with torch.inference_mode():
        model.eval()
        for x_batch, y_batch in test_dataloader:
            logits = model(x_batch)
            energy = energy_score(logits)

    accuracy = get_accuracy(model, test)
    accuracies.append(accuracy)
    energies.append(energy)
    torch.save(model.state_dict(), f"./fold-{fold + 1}.pt")

pd.DataFrame({"accuracy": accuracies, "energy": energies}).to_csv("./data.csv")
