import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


from sys import argv
from torchvision.models import resnet50 as resnet
from datasets import portraits
from functions import seed
from functions import get_device
from functions import flush


def main():
    if len(argv) != 3:
        return print("You need to specify two model addresses")

    first_model = argv[1]
    second_model = argv[2]

    device = get_device()
    dataset, num_classes = portraits()
    image, label = random.choice(dataset)
    image = image.unsqueeze(0)

    model = resnet().to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(first_model, map_location=torch.device("cpu"), weights_only=True)
    )
    fig1, logits1 = evaluate(model, image, label, num_classes)

    model = resnet().to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(second_model, map_location=torch.device("cpu"), weights_only=True)
    )
    fig2, logits2 = evaluate(model, image, label, num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(list(range(num_classes)), logits1.detach().numpy()[0], color="skyblue")
    axes[0].set_xlabel("Classes")
    z = logits1.std() / logits1.mean().abs()
    axes[0].set_ylabel("Logits")
    axes[0].set_title(f"Crispy {round(z.item(), 2)}")

    axes[1].bar(list(range(num_classes)), logits2.detach().numpy()[0], color="salmon")
    axes[1].set_xlabel("Classes")
    axes[1].set_ylabel("Logits")
    z = logits2.std() / logits2.mean().abs()
    axes[1].set_title(f"Regular {round(z.item(), 2)}")
    plt.tight_layout()
    plt.savefig(f"{int(time.time())}.png")
    plt.close()


def evaluate(model, image, label, num_classes):
    model.eval()
    logits = model(image)
    predicted_index = torch.argmax(logits).item()

    if predicted_index == label.item():
        flush("✅ Predicted correctly")
    else:
        flush("❌ Predicted wrong")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(list(range(num_classes)), logits.detach().numpy()[0], color="skyblue")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Logits")
    ax.set_title("Class Probabilities")
    return fig, logits


if __name__ == "__main__":
    main()
