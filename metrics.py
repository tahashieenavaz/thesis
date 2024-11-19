from functions import get_device
import torch


def get_accuracy(model, test) -> float:
    device = get_device()
    test_dataloader = torch.utils.data.DataLoader(test, shuffle=False)
    total = 0
    correct = 0
    with torch.inference_mode():
        for x_batch, y_batch in test_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total * 100
