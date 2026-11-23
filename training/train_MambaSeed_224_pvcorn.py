"""Train MambaSeed-224 on pvcorn dataset (4 classes, low-FLOPs).

- 模型名: mamba_seed_224
- 输入分辨率: 224x224
- 数据集: dataset/pvcorn (train / validation)
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import timm

# 确保可以从本工程的 model 目录导入 mambaseed 定义
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model"))
from mambaseed import mamba_seed_224  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Train MambaSeed-224 on pvcorn dataset")
    parser.add_argument('--data-root', default=str(REPO_ROOT / "dataset/pvcorn"), type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    # parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--img-size', default=224, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--output', default=str(REPO_ROOT / "outputs/pvcorn"), type=str)
    return parser.parse_args()


def build_mambaseed_224(num_classes: int, image_size: int = 224) -> nn.Module:
    model = timm.create_model(
        "mamba_seed_224",
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def build_dataloaders(data_root, img_size, batch_size, num_workers):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'validation')

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, len(train_ds.classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_num = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += images.size(0)

    return total_loss / total_num, total_correct / total_num


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_num += images.size(0)

    return total_loss / total_num, total_correct / total_num


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    wandb.init(
        project='pvcorn_MambaSeed_224',
        name='MambaSeed_224_from_scratch_pvcorn',
        dir=str(REPO_ROOT),
        config={
            'data_root': args.data_root,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'img_size': args.img_size,
            'num_workers': args.num_workers,
            'num_classes': 4,
            'model': 'mamba_seed_224',
        },
    )

    train_loader, val_loader, num_classes = build_dataloaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers
    )

    # pvcorn 为 4 类
    num_classes = 4
    model = build_mambaseed_224(num_classes=num_classes, image_size=args.img_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
        })

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(
                args.output,
                f"best_mamba_seed_224_pvcorn_{timestamp}.pth",
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> New best val acc: {best_acc:.4f}, checkpoint saved to {ckpt_path}.')

        scheduler.step()

    wandb.finish()


if __name__ == '__main__':
    main()
