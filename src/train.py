"""
Train the hybrid EfficientNet-B4 + ViT-B/16 disease classifier.

    python src/train.py [--epochs N] [--batch-size N]

Saves the best checkpoint (lowest val loss) to models/best_model.pt and a
per-epoch metrics log to results/training_log.csv + results/training_curves.png.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (BATCH_SIZE, CLASS_NAMES, EARLY_STOPPING_PATIENCE, EPOCHS,
                     LEARNING_RATE, MODELS_DIR, RESULTS_DIR, SEED, WEIGHT_DECAY)
from data import get_dataloaders
from model import HybridFishDiseaseModel

CUTMIX_PROB = 0.3
CUTMIX_ALPHA = 1.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = CUTMIX_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)

    _, _, h, w = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)

    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (h * w))

    return mixed, labels, labels[perm], lam_adjusted


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            use_cutmix = train and np.random.rand() < CUTMIX_PROB
            if use_cutmix:
                images, labels_a, labels_b, lam = cutmix_batch(images, labels)

            if train:
                optimizer.zero_grad()

            logits = model(images)

            if use_cutmix:
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("[warn] no GPU found -- this hybrid model will be slow to train on CPU. "
              "Use Colab (Runtime > Change runtime type > GPU) for a real run.")

    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size)
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = HybridFishDiseaseModel(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        elapsed = time.time() - start

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                          "val_loss": val_loss, "val_acc": val_acc, "lr": lr})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, MODELS_DIR / "best_model.pt")
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    with open(RESULTS_DIR / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    plot_training_curves(log_rows)
    print(f"\nBest val_loss={best_val_loss:.4f}. Checkpoint: {MODELS_DIR / 'best_model.pt'}")
    print("Run `python src/evaluate.py` for held-out test metrics.")


def plot_training_curves(log_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [r["train_loss"] for r in log_rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in log_rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r["train_acc"] for r in log_rows], label="train")
    axes[1].plot(epochs, [r["val_acc"] for r in log_rows], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
