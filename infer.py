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
    label = label.item()

    # first model evaluation
    model = resnet().to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(first_model, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()
    logits1 = model(image)
    predicted_index = torch.argmax(logits1).item()
    flush(
        "✅ Predicted correctly" if predicted_index == label else "❌ Predicted wrong"
    )

    # second model evaluation
    model = resnet().to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(second_model, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()
    logits2 = model(image)
    predicted_index = torch.argmax(logits2).item()
    flush(
        "✅ Predicted correctly" if predicted_index == label else "❌ Predicted wrong"
    )

    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(list(range(num_classes)), logits1.detach().numpy()[0], color="skyblue")
    axes[0].set_xlabel("Classes")
    axes[0].set_ylabel("Logits")
    axes[0].set_title(f"Crispy {round(logits1.std().item(), 2)}")

    axes[1].bar(list(range(num_classes)), logits2.detach().numpy()[0], color="salmon")
    axes[1].set_xlabel("Classes")
    axes[1].set_ylabel("Logits")
    axes[1].set_title(f"Regular {round(logits2.std().item(), 2)}")
    plt.tight_layout()
    plt.savefig(f"{int(time.time())}.png")
    plt.close()


if __name__ == "__main__":
    main()
