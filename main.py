"""Train yoga pose classifiers (complex & real-valued backbones).

Usage
-----
python main.py --root YogaPoses --arch resnet50 --epochs 50 --batch_size 8 --lr 3e-4 --modelName resnet50

Args (highlights)
-----------------
--root: Dataset root with class subfolders.
--arch: resnet18|resnet50|alexnet|vgg11|vgg16|stannet
--image_size: Input size (default 224)
--val_ratio: Validation split ratio (default 0.2)
--batch_size, --epochs, --lr, --weight_decay, --num_workers
--lamb_phase: Phase-loss weight (float, default 0.05)
--strong_aug: Enable stronger spatial augments (flag)
--seed: RNG seed (default 42)
--out_dir: Output directory for checkpoints (default saved_models/)
--modelName: Base name for saved checkpoints (default best_model)

Notes
-----
- Auto-detects CUDA if available.
- Uses HSV→complex conversion from utils.ToHSV/ToComplex.
"""

import os
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from einops import rearrange

# iHSV and complex-image construction + progress bar from your utils.py
from utils import ToHSV, ToComplex, progress_bar  # uses rgb_to_hsv_mine and iHSV stacking internally

# Complex backbones from your repo that expect 3 complex input channels
import networks as models
import stanNet as stan_models


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int, strong_aug: bool):
    aug = []
    if strong_aug:
        aug = [
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
        ]

    train_tf = T.Compose([
        T.ToTensor(),
        T.Resize(image_size + 8),
        T.CenterCrop(image_size),
        *aug,
        ToHSV(),
        ToComplex(),
    ])

    val_tf = T.Compose([
        T.ToTensor(),
        T.Resize(image_size),
        T.CenterCrop(image_size),
        ToHSV(),
        ToComplex(),
    ])

    return train_tf, val_tf


def stratified_indices(targets, val_ratio=0.2, seed=42):
    rnd = random.Random(seed)
    per_class = {}
    for idx, t in enumerate(targets):
        per_class.setdefault(int(t), []).append(idx)

    train_idx, val_idx = [], []
    for cls, idxs in per_class.items():
        rnd.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]

    return train_idx, val_idx


def criterion_phase(y, theta, num_classes: int):
    # y: (B,) long; theta: (B, C)
    y_oh = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    return (y_oh * theta).mean()


def build_model(arch: str, num_classes: int):
    a = arch.lower()
    if a == "resnet18":
        return models.resnet18(num_classes=num_classes)
    if a == "resnet50":
        return models.resnet50(num_classes=num_classes)
    if a == "alexnet":
        return models.AlexNet(num_classes=num_classes)
    if a == "vgg11":
        return models.VGG('Vgg11', num_classes=num_classes)
    if a == "vgg16":
        return models.VGG('Vgg16', num_classes=num_classes)
    if a == "stannet":
        return stan_models.stanNet_complex(num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")

def save_checkpoint(state, out_dir: Path, filename: str = "best"):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{filename}.pth"
    torch.save(state, path)
    return str(path)

def train_one_epoch(model, loader, optimizer, ce_loss, lamb_phase, device, num_classes: int):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for bid, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        z = model(x)  # complex logits (B, C, 1, 1)
        z = rearrange(z, "b c h w -> b (c h w)")
        assert torch.is_complex(z), "Model must output complex logits (complex dtype)"

        mag, pha = z.abs(), z.angle()
        loss_mag = ce_loss(mag, y)
        loss_pha = criterion_phase(y, pha, num_classes)

        # Use small phase weight; set --lamb_phase 0.0 to disable during debugging
        loss = loss_mag - lamb_phase * loss_pha

        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        total += y.size(0)
        pred = mag.argmax(dim=1)
        correct += (pred == y).sum().item()

        progress_bar(bid, len(loader),
                     'Loss: %.3f | Acc: %.2f%% (%d/%d)' %
                     (run_loss / (bid + 1), 100.0 * correct / total, correct, total))

    return run_loss / max(1, len(loader)), correct / max(1, total)

@torch.no_grad()
def evaluate(model, loader, ce_loss, device, num_classes: int):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    for bid, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        z = model(x)
        z = rearrange(z, "b c h w -> b (c h w)")
        assert torch.is_complex(z), "Model must output complex logits (complex dtype)"

        mag = z.abs()
        loss = ce_loss(mag, y)  # evaluate on CE(magnitude) for cleaner tracking
        run_loss += float(loss)
        total += y.size(0)
        pred = mag.argmax(dim=1)
        correct += (pred == y).sum().item()

        progress_bar(bid, len(loader),
                     'Val Loss: %.3f | Val Acc: %.2f%% (%d/%d)' %
                     (run_loss / (bid + 1), 100.0 * correct / total, correct, total))

    return run_loss / max(1, len(loader)), correct / max(1, total)

def main():
    p = argparse.ArgumentParser(description="Yoga pose classification (single-folder) with complex CNNs")
    p.add_argument("--root", type=str, required=True, help="Path to yogaPose folder with class subfolders")
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "resnet50", "alexnet", "vgg11", "vgg16", "stannet"],)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)  # slightly higher to help small datasets
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lamb_phase", type=float, default=0.05)  # small default; set 0.0 to disable
    p.add_argument("--strong_aug", action="store_true", help="Enable flip/rotation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="saved_models/")
    p.add_argument("--modelName", type=str, default="best_model")
    p.add_argument("--num_params", action="store_true", help="Print model parameter count")

    args = p.parse_args()

    # Validate root path exists
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: Dataset path does not exist: {args.root}")
        return

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_tf, val_tf = build_transforms(args.image_size, strong_aug=args.strong_aug)

    # Build a base dataset to compute a stratified split
    base = torchvision.datasets.ImageFolder(root=args.root, transform=None)
    classes = base.classes
    targets = base.targets
    train_idx, val_idx = stratified_indices(targets, val_ratio=args.val_ratio, seed=args.seed)

    # Two views with different transforms over the same files
    train_set = torchvision.datasets.ImageFolder(root=args.root, transform=train_tf)
    val_set = torchvision.datasets.ImageFolder(root=args.root, transform=val_tf)

    train_sub = Subset(train_set, train_idx)
    val_sub = Subset(val_set, val_idx)

    num_classes = len(classes)

    print(f"Classes ({num_classes}): {classes}")

    pin = True if device.type == "cuda" else False

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = build_model(args.arch, num_classes=num_classes).to(device)

    # print model parameter count if requested
    if args.num_params:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameter count for {args.arch}: {param_count}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_dir = Path(args.out_dir)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, ce_loss,
                                          args.lamb_phase, device, num_classes)
        va_loss, va_acc = evaluate(model, val_loader, ce_loss, device, num_classes)

        if va_acc > best_acc:
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "arch": args.arch.lower(), # normalize to lowercase
                "classes": classes,
                "num_classes": num_classes,
                "image_size": args.image_size,
            }

            path = save_checkpoint(ckpt, out_dir, args.modelName)
            print(f"Saved best checkpoint to: {path}")
            best_acc = va_acc

        print(f"Epoch {epoch + 1}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} best={best_acc:.4f}")

if __name__ == "__main__":
    main()
