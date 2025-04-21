import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Tuple, List
from common import DEVICE


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer
                    ) -> Tuple[float, float]:
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        count += imgs.size(0)
    return total_loss/count, correct/count


def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module
             ) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            count += imgs.size(0)
    return total_loss/count, correct/count


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int
                ) -> dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = validate(model, val_loader, criterion)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"Epoch {_+1}/{epochs} - "
              f"Train Loss: {tl:.4f}, Train Acc: {ta:.4f} - "
              f"Val Loss: {vl:.4f}, Val Acc: {va:.4f}")
    return history
