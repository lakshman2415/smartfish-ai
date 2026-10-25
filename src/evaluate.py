"""
Evaluate the best checkpoint on the held-out test split (never seen during
training or early stopping).

    python src/evaluate.py

Writes results/classification_report.txt, results/confusion_matrix.png,
results/roc_curves.png and results/metrics.json.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from config import CLASS_NAMES, MODELS_DIR, RESULTS_DIR
from data import FishDiseaseDataset, get_transforms
from model import HybridFishDiseaseModel


def load_model(device: torch.device) -> HybridFishDiseaseModel:
    checkpoint = torch.load(MODELS_DIR / "best_model.pt", map_location=device)
    model = HybridFishDiseaseModel(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs = model.predict_proba(images).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_labels.append(labels.numpy())

    return (np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs))


def specificity_per_class(cm: np.ndarray) -> list[float]:
    total = cm.sum()
    specs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return specs


def plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    n_classes = len(CLASS_NAMES)
    y_true_onehot = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        auc = roc_auc_score(y_true_onehot[:, i], y_proba[:, i])
        aucs.append(auc)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    return float(np.mean(aucs))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(device)
    test_ds = FishDiseaseDataset("test", get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_ds)}")

    y_true, y_pred, y_proba = collect_predictions(model, test_loader, device)

    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n" + report_text)
    (RESULTS_DIR / "classification_report.txt").write_text(report_text)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    specs = specificity_per_class(cm)
    macro_auc = plot_roc_curves(y_true, y_proba)
    accuracy = float((y_true == y_pred).mean())

    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "specificity_macro": float(np.mean(specs)),
        "roc_auc_macro": macro_auc,
        "specificity_per_class": dict(zip(CLASS_NAMES, specs)),
        "n_test_samples": len(y_true),
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTest accuracy: {accuracy:.4f} | macro F1: {f1:.4f} | macro ROC-AUC: {macro_auc:.4f}")
    print(f"Full results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
