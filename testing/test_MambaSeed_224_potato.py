"""MambaSeed-224 Low-FLOPs Test Script (for Potato Dataset)

在马铃薯（Potato）病害分类数据集上测试低 FLOPs 版本 MambaSeed224 模型。
- 模型名: mamba_seed_224
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb
import timm

# 确保可以从本工程的 model 目录导入 mambaseed 定义
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model"))
from mambaseed import mamba_seed_224  # noqa: F401


def build_mambaseed_224(num_classes: int, image_size: int = 224) -> nn.Module:
    model = timm.create_model(
        "mamba_seed_224",
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def get_test_loader(data_root, image_size=224, batch_size=64, num_workers=4, split="test"):
    test_dir = os.path.join(data_root, split)

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return test_loader, len(test_dataset.classes)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def get_preds_and_targets(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, targets_batch in loader:
            images = images.to(device, non_blocking=True)
            targets_batch = targets_batch.to(device, non_blocking=True)
            outputs = model(images)
            _, p = outputs.max(1)
            preds.extend(p.cpu().numpy().tolist())
            targets.extend(targets_batch.cpu().numpy().tolist())
    return targets, preds


def compute_metrics(y_true, y_pred, num_classes):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity = np.nan_to_num(tp / (tp + fn)).mean()
        specificity = np.nan_to_num(tn / (tn + fp)).mean()
        precision = np.nan_to_num(tp / (tp + fp)).mean()
        f1 = np.nan_to_num(2 * precision * sensitivity / (precision + sensitivity)).mean()
        accuracy = float(tp.sum() / cm.sum())
    return {"sensitivity": float(sensitivity), "specificity": float(specificity), "f1": float(f1), "accuracy": float(accuracy)}


def main():
    # 和训练脚本保持一致（仅数据路径与权重目录不同）
    data_root = str(REPO_ROOT / "dataset/potato")
    image_size = 224
    batch_size = 64
    num_workers = 4
    split = "test"  # 如果没有单独 test，可改为 "validation"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="potato_MambaSeed_224",
        name=f"MambaSeed_224_potato_test_{split}",
        dir=str(REPO_ROOT),
        config={
            "data_root": data_root,
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "split": split,
            "model": "mamba_seed_224",
        },
    )

    test_loader, num_classes = get_test_loader(
        data_root, image_size=image_size, batch_size=batch_size,
        num_workers=num_workers, split=split,
    )
    # Force number of classes to 9 for Potato dataset
    num_classes = 9

    model = build_mambaseed_224(num_classes=num_classes, image_size=image_size).to(device)

    criterion = nn.CrossEntropyLoss()

    # 加载最新时间戳的 best 权重（Potato 专属目录）
    weights_dir = str(REPO_ROOT / "outputs/potato")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    candidates = [
        f for f in os.listdir(weights_dir)
        if f.startswith("best_mamba_seed_224_potato_") and f.endswith(".pth")
    ]
    if not candidates:
        raise FileNotFoundError(f"No best weights found in {weights_dir}")

    candidates.sort()
    load_path = os.path.join(weights_dir, candidates[-1])

    state_dict = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded weights from {load_path}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    y_true, y_pred = get_preds_and_targets(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)

    print(f"[Potato-{split}] Loss: {test_loss:.4f}  Acc: {metrics['accuracy']:.4f} \n"
          f"Sensitivity: {metrics['sensitivity']:.4f}  Specificity: {metrics['specificity']:.4f}  F1: {metrics['f1']:.4f}")

    wandb.log({
        "test_loss": test_loss,
        "test_acc": metrics['accuracy'],
        "test_sensitivity": metrics['sensitivity'],
        "test_specificity": metrics['specificity'],
        "test_f1": metrics['f1'],
        "split": split,
    })

    wandb.finish()


if __name__ == "__main__":
    main()
