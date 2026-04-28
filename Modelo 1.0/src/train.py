import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import os

from src.utils import save_checkpoint
from src.metrics import compute_metrics
from src.metrics import plot_training_history

def train(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)

    best_loss = float("inf")
    patience = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    model.to(config.DEVICE)

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        y_true, y_pred = [], []

        for images, labels in tqdm(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        train_metrics = compute_metrics(y_true, y_pred)

        val_loss, val_metrics = evaluate(model, val_loader, criterion, config)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, os.path.join(config.MODEL_DIR, "best_model.pth"))
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                break

    with open(os.path.join(config.METRIC_DIR, "history.json"), "w") as f:
        json.dump(history, f)
    
    plot_training_history(history, config.PLOT_DIR)
    return history


def evaluate(model, loader, criterion, config):
    model.eval()
    loss_total = 0

    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_total += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, y_probs)
    return loss_total, metrics