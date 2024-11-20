import torch
import numpy as np
import pandas as pd

from datasets import portraits
from functions import get_device
from functions import load_settings
from functions import flush
from functions import seed
from functions import create_folder
from metrics import get_accuracy
from scores import energy_score
from losses import LogitNormLoss
from torchvision.models import resnet18 as resnet
from sklearn.model_selection import KFold


seed()
dataset = portraits()
device = get_device()
settings = load_settings()
kf = KFold(n_splits=settings.folds)

thresholds = []
accuracies = []
create_folder("./models")

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, test_idx)

    model = resnet().to(device)

    criterion = LogitNormLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [criterion.temperature],
        settings.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)

    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=settings.batch_size
    )

    flush(f"fold {fold + 1} was started")
    for epoch in range(settings.epochs):
        train_dataloader = torch.utils.data.DataLoader(
            train, shuffle=True, batch_size=settings.batch_size
        )

        model.train()
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        flush(f"\tepoch {epoch + 1} was finished")
    # epochs end
    flush(f"fold {fold + 1} was finished")

    scores = torch.tensor([], device=device)
    with torch.inference_mode():
        model.eval()
        for x_batch, y_batch in test_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            batch_scores = energy_score(logits)
            scores = torch.cat((batch_scores.flatten(), scores.flatten()))
            flush(scores)

    accuracy = get_accuracy(model, test)
    accuracies.append(accuracy)

    threshold = torch.quantile(scores, 0.95)
    thresholds.append(threshold.item())

    torch.save(model.state_dict(), f"./models/fold-{fold + 1}.pt")


results_df = pd.DataFrame({"accuracy": accuracies, "threshold": thresholds})
results_df.index.name = "id"
results_df.to_csv("./data.csv")
