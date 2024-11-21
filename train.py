import torch
import numpy as np
import pandas as pd

from datasets import portraits
from functions import get_device
from functions import load_settings
from functions import flush
from functions import seed
from functions import create_folder
from functions import build_resnet
from functions import build_optimizer
from metrics import get_accuracy
from metrics import get_kappa
from metrics import get_f1
from scores import energy_score
from scores import get_scores
from losses import MarginEnhancedLogitNormLoss
from sklearn.model_selection import KFold
from copy import deepcopy


seed()
dataset, num_classes = portraits()
device = get_device()
settings = load_settings()
kf = KFold(n_splits=settings.folds)
create_folder("./models")

thresholds = []
accuracies = []
f1s = []
kappas = []
temperatures = []
margins = []


flush(f"The dataset has {num_classes} classes")

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    fold_model = None
    best_accuracy = float("-inf")

    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, test_idx)
    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=settings.batch_size
    )

    criterion = MarginEnhancedLogitNormLoss()
    model = build_resnet(num_classes)
    optimizer, step = build_optimizer(
        model=model, criterion=criterion, lr=settings.lr, theta=settings.theta
    )

    flush(f"fold {fold + 1} was started")
    for epoch in range(settings.epochs):
        train_dataloader = torch.utils.data.DataLoader(
            train, shuffle=True, batch_size=settings.batch_size
        )
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        step(epoch)

        accuracy = get_accuracy(model, test_dataloader)
        if accuracy > best_accuracy:
            fold_model = deepcopy(model)
            best_accuracy = accuracy

        flush(f"\tepoch {epoch + 1} was finished with {epoch_loss}")
    # epochs end
    flush(f"fold {fold + 1} was finished")

    threshold = torch.quantile(
        get_scores(fold_model, energy_score, test_dataloader), 0.95
    ).item()
    kappa = get_kappa(fold_model, test_dataloader)
    f1 = get_f1(fold_model, test_dataloader)

    accuracies.append(best_accuracy)
    thresholds.append(threshold)
    kappas.append(kappa)
    f1s.append(f1)
    temperatures.append(criterion.temperature.item())
    margins.append(criterion.margin.item())

    torch.save(fold_model.state_dict(), f"./models/fold-{fold + 1}.pt")
    flush(
        f"\n\taccuracy: {best_accuracy}, threshold: {threshold}, f1: {f1}, kappa: {kappa}"
    )


results_df = pd.DataFrame(
    {
        "accuracy": accuracies,
        "threshold": thresholds,
        "kappa": kappas,
        "f1": f1s,
        "margin": margins,
        "temperature": temperatures,
    }
)
results_df.index.name = "id"
results_df.to_csv("./data.csv")
