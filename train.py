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
from losses import LogitNormLoss
from sklearn.model_selection import KFold


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
    criterion = LogitNormLoss()
    cnn_params = [
        param for name, param in model.named_parameters() if not name.startswith("fc")
    ]
    fc_params = [
        param for name, param in model.named_parameters() if name.startswith("fc")
    ]
    optimizer = torch.optim.SGD(
        [
            {"params": cnn_params, "lr": settings.lr},
            {"params": fc_params, "lr": settings.lr * 20},
        ],
        momentum=0.9,
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
        flush(f"\tepoch {epoch + 1} was finished with {epoch_loss}")
    # epochs end
    flush(f"fold {fold + 1} was finished")

    scores = torch.tensor([], device=device)
    model.eval()
    with torch.inference_mode():
        for x_batch, y_batch in test_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            batch_scores = energy_score(logits)
            scores = torch.cat((batch_scores.flatten(), scores.flatten()))

    threshold = torch.quantile(scores, 0.95).item()
    accuracy = get_accuracy(model, test_dataloader)
    kappa = get_kappa(model, test_dataloader)
    f1 = get_f1(model, test_dataloader)

    accuracies.append(accuracy)
    thresholds.append(threshold)
    kappas.append(kappa)
    f1s.append(f1)

    torch.save(model.state_dict(), f"./models/fold-{fold + 1}.pt")
    flush(f"\n\taccuracy: {accuracy}, threshold: {threshold}, f1: {f1}, kappa: {kappa}")


results_df = pd.DataFrame(
    {"accuracy": accuracies, "threshold": thresholds, "kappa": kappa, "f1": f1s}
)
results_df.index.name = "id"
results_df.to_csv("./data.csv")
