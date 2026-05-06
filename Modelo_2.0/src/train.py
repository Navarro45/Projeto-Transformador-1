import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def train_model(model, train_loader, val_loader, config):
    model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    train_losses = []
    val_losses = []

    best_val = float("inf")
    patience = 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train = 0

        for images, labels in tqdm(train_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train += loss.item()

        model.eval()
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val += loss.item()

        train_losses.append(total_train)
        val_losses.append(total_val)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {total_train:.4f}")
        print(f"Val Loss: {total_val:.4f}")

        scheduler.step(total_val)

        if total_val < best_val:
            best_val = total_val
            patience = 0
        else:
            patience += 1

        if patience >= config.EARLY_STOPPING_PATIENCE:
            print("⛔ Early stopping")
            break

    # ================= PLOT =================
    os.makedirs(config.PATHS["plots"], exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(config.PATHS["plots"], "loss_curve.png"))
    plt.close()