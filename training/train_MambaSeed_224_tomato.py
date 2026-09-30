"""MambaSeed-224 Low-FLOPs Training Script (for Tomato Dataset)

在番茄病害分类数据集上训练低 FLOPs 版本 MambaSeed224 模型。
- 模型名: mamba_seed_224
"""

import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

# 确保可以从本工程的 model 目录导入 mambaseed 定义
sys.path.append("/root/autodl-tmp/MambaSeed_224/model")
from mambaseed import mamba_seed_224  # noqa: F401

def build_mambaseed_224(num_classes: int, image_size: int = 224) -> nn.Module:
    model = mamba_seed_224(num_classes=num_classes)
    return model

def get_dataloaders(data_root, image_size=224, batch_size=64, num_workers=4):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "validation")

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(train_dataset.classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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


def main():
    data_root = "/root/autodl-tmp/data/tomato"
    image_size = 224
    batch_size = 64
    num_workers = 4
    num_epochs = 50
    # lr = 3e-4
    lr = 4e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="tomato_MambaSeed_224",
        name="MambaSeed_224_tomato",
        dir="/root/autodl-tmp/MambaSeed_224",
        config={
            "data_root": data_root,
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "num_epochs": num_epochs,
            "lr": lr,
            "model": "mamba_seed_224",
        },
    )

    train_loader, val_loader, num_classes = get_dataloaders(
        data_root, image_size=image_size, batch_size=batch_size, num_workers=num_workers
    )

    model = build_mambaseed_224(num_classes=num_classes, image_size=image_size).to(device)

    print(f"Current expand: {model.expand}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将权重统一保存到 Tomato 专属目录
    weights_dir = "/root/autodl-tmp/MambaSeed_224/output_tomato_MambaSeed_224"
    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(
                weights_dir,
                f"best_mamba_seed_224_tomato_{timestamp}.pth",
            )
            torch.save(model.state_dict(), save_path)
            print(f"==> New best MambaSeed-224 model saved at {save_path}, acc={best_acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
