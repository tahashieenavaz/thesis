import torch
import numpy as np
import pandas as pd

from datasets import portraits
from functions import get_device
from functions import load_settings
from functions import flush
from functions import seed
from functions import create_folder
from functions import build_model
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


flush(f"The dataset has {num_classes} classes")

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, test_idx)

    model = build_model(num_classes)

    fold_model = None
    best_accuracy = float("-inf")

    criterion = MarginEnhancedLogitNormLoss()
    cnn_params = [
        param for name, param in model.named_parameters() if not name.startswith("fc")
    ]
    fc_params = [
        param for name, param in model.named_parameters() if name.startswith("fc")
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": cnn_params, "lr": settings.lr},
            {"params": fc_params, "lr": settings.lr * 20},
            {"params": [criterion.temperature, criterion.margin], "lr": settings.lr},
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, gamma=settings.gamma, step_size=settings.step_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=settings.batch_size
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
        scheduler.step()

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

    torch.save(fold_model.state_dict(), f"./models/fold-{fold + 1}.pt")
    flush(
        f"\n\taccuracy: {best_accuracy}, threshold: {threshold}, f1: {f1}, kappa: {kappa}"
    )


results_df = pd.DataFrame(
    {"accuracy": accuracies, "threshold": thresholds, "kappa": kappas, "f1": f1s}
)
results_df.index.name = "id"
results_df.to_csv("./data.csv")
