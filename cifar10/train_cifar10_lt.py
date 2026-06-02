"""
CIFAR-10 LT + EDM Synthetic 학습 스크립트
==========================================
실험 계획:
  실험 1: 부족분의 25% 채우기
  실험 2: 부족분의 50% 채우기
  실험 3: 부족분의 75% 채우기
  실험 4: 부족분의 100% 채우기 (5,000개로 완성)

사용 예시:
  # Real only (기준점)
  python train_cifar10_lt.py \
    --mode real \
    --output_dir ./outputs/exp_cifar10_real

  # 실험 1: 부족분의 25% 채우기
  python train_cifar10_lt.py \
    --mode synthetic \
    --fill_ratio 0.25 \
    --checkpoint ./pytorch-cifar/checkpoint/ckpt.pth \
    --output_dir ./outputs/exp_cifar10_synth_25

  # 실험 2: 부족분의 50% 채우기
  python train_cifar10_lt.py \
    --mode synthetic \
    --fill_ratio 0.50 \
    --checkpoint ./pytorch-cifar/checkpoint/ckpt.pth \
    --output_dir ./outputs/exp_cifar10_synth_50

  # 실험 3: 부족분의 75% 채우기
  python train_cifar10_lt.py \
    --mode synthetic \
    --fill_ratio 0.75 \
    --checkpoint ./pytorch-cifar/checkpoint/ckpt.pth \
    --output_dir ./outputs/exp_cifar10_synth_75

  # 실험 4: 부족분의 100% 채우기
  python train_cifar10_lt.py \
    --mode synthetic \
    --fill_ratio 1.0 \
    --checkpoint ./pytorch-cifar/checkpoint/ckpt.pth \
    --output_dir ./outputs/exp_cifar10_synth_100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

# ── 전처리 (kuangliu 기준) ────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])


# ── Dataset ───────────────────────────────────────────────────────

class LTSubset(Dataset):
    """CIFAR-10 LT subset. label을 tensor로 반환."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, torch.tensor(label, dtype=torch.long)


class SyntheticDataset(Dataset):
    """EDM으로 생성된 이미지 로드."""
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ── 데이터 로드 ───────────────────────────────────────────────────

def load_cifar10_lt(
    longtail_stats_path: str,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 42,
):
    """CIFAR-10 LT 로드."""
    with open(longtail_stats_path) as f:
        stats = json.load(f)

    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=TRAIN_TRANSFORM)

    random.seed(seed)
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(full_trainset):
        label_to_indices[label].append(idx)

    selected = []
    label_counts = Counter()
    for cls_idx, info in stats['class_distribution'].items():
        n = info['n_samples']
        indices = label_to_indices[int(cls_idx)]
        sampled = random.sample(indices, min(n, len(indices)))
        selected.extend(sampled)
        label_counts[int(cls_idx)] = len(sampled)

    trainset = LTSubset(full_trainset, selected)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=TEST_TRANSFORM)

    logger.info("CIFAR-10 LT train: %d samples", len(trainset))
    logger.info("CIFAR-10 test: %d samples", len(testset))
    for i in range(10):
        logger.info("  class %d (%s): %d개", i, CIFAR10_CLASSES[i], label_counts[i])

    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=100,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, label_counts, trainset


def load_synthetic_with_fill_ratio(
    synth_data_root: str,
    label_counts: Counter,
    fill_ratio: float = 1.0,
    max_per_class: int = 5000,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """부족분의 fill_ratio만큼 synthetic 로드."""
    random.seed(seed)
    root = Path(synth_data_root)
    selected = []

    for cls_idx in range(10):
        real_count = label_counts.get(cls_idx, 0)
        deficit = max(0, max_per_class - real_count)
        n_to_add = int(deficit * fill_ratio)

        if n_to_add == 0:
            continue

        class_dir = root / f"class{cls_idx}"
        if not class_dir.exists():
            logger.warning("class %d synthetic 폴더 없음: %s", cls_idx, class_dir)
            continue

        all_imgs = sorted(class_dir.glob("*.png"))
        if not all_imgs:
            all_imgs = sorted(class_dir.glob("*.jpg"))

        if len(all_imgs) == 0:
            logger.warning("class %d synthetic 이미지 없음", cls_idx)
            continue

        if len(all_imgs) >= n_to_add:
            chosen = random.sample(all_imgs, n_to_add)
        else:
            chosen = random.choices(all_imgs, k=n_to_add)

        selected.extend([(str(p), cls_idx) for p in chosen])
        logger.info("  class %d (%s): real %d + synthetic %d (deficit %d의 %.0f%%)",
                    cls_idx, CIFAR10_CLASSES[cls_idx],
                    real_count, len(chosen), deficit, fill_ratio * 100)

    logger.info("Total synthetic added: %d", len(selected))
    return selected


# ── 모델 ──────────────────────────────────────────────────────────

def build_resnet50(device: str) -> nn.Module:
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str) -> nn.Module:
    logger.info("Loading checkpoint: %s", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device)
    if 'net' in state_dict:
        new_state = OrderedDict()
        for k, v in state_dict['net'].items():
            k = k.replace('module.', '')
            k = k.replace('shortcut', 'downsample')
            k = k.replace('linear', 'fc')
            new_state[k] = v
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(state_dict)
    return model


# ── Validation ────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, test_loader, device, label_counts):
    model.eval()
    correct = total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        for p, l in zip(pred.cpu(), labels.cpu()):
            class_total[int(l)] += 1
            if p == l:
                class_correct[int(l)] += 1

    overall = correct / total * 100

    median = sorted(label_counts.values())[len(label_counts) // 2]
    head_c = head_t = tail_c = tail_t = 0
    for i in range(10):
        if label_counts.get(i, 0) >= median:
            head_c += class_correct[i]
            head_t += class_total[i]
        else:
            tail_c += class_correct[i]
            tail_t += class_total[i]

    head_acc = head_c / head_t * 100 if head_t > 0 else 0
    tail_acc = tail_c / tail_t * 100 if tail_t > 0 else 0

    class_acc = {}
    for i in range(10):
        acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        class_acc[f"class{i}_{CIFAR10_CLASSES[i]}"] = round(acc, 2)

    return {
        "overall": round(overall, 2),
        "head": round(head_acc, 2),
        "tail": round(tail_acc, 2),
        "class_acc": class_acc,
    }


# ── 학습 루프 ─────────────────────────────────────────────────────

def train(model, train_loader, test_loader, label_counts,
          num_epochs=200, base_lr=0.1, device="cuda", output_dir="./outputs"):

    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log = []
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        scheduler.step()
        avg_loss = total_loss / total_samples
        acc = validate(model, test_loader, device, label_counts)

        if acc["overall"] > best_acc:
            best_acc = acc["overall"]
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")

        log.append({"epoch": epoch, "loss": round(avg_loss, 4), **acc})

        logger.info(
            "[Epoch %3d/%d] loss=%.4f | overall=%.2f%% | head=%.2f%% | tail=%.2f%% | best=%.2f%%",
            epoch, num_epochs, avg_loss,
            acc["overall"], acc["head"], acc["tail"], best_acc,
        )

    logger.info("Training complete. Best Overall: %.2f%%", best_acc)
    with open(f"{output_dir}/train_log.json", "w") as f:
        json.dump(log, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real", "synthetic"], required=True)
    parser.add_argument("--longtail_stats", type=str,
                        default="./dataset/cifar10_longtail/longtail_stats.json")
    parser.add_argument("--synth_data_root", type=str,
                        default="./dataset/synth_cifar10_edm")
    parser.add_argument("--fill_ratio", type=float, default=1.0,
                        help="부족분의 몇 %를 채울지 (0.25/0.5/0.75/1.0)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/exp_cifar10")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info("Mode: %s | Device: %s", args.mode, args.device)

    train_loader, test_loader, label_counts, real_ds = load_cifar10_lt(
        longtail_stats_path=args.longtail_stats,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_resnet50(args.device)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint, args.device)
        logger.info("Fine-tuning from: %s", args.checkpoint)

    if args.mode == "synthetic":
        logger.info("fill_ratio=%.2f (부족분의 %.0f%%)", args.fill_ratio, args.fill_ratio * 100)
        synth_samples = load_synthetic_with_fill_ratio(
            synth_data_root=args.synth_data_root,
            label_counts=label_counts,
            fill_ratio=args.fill_ratio,
            seed=args.seed,
        )
        synth_ds = SyntheticDataset(synth_samples, transform=TRAIN_TRANSFORM)
        combined_ds = ConcatDataset([real_ds, synth_ds])
        train_loader = DataLoader(combined_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        logger.info("Combined: Real %d + Synthetic %d = %d",
                    len(real_ds), len(synth_ds), len(combined_ds))

    train(model, train_loader, test_loader, label_counts,
          num_epochs=args.num_epochs, base_lr=args.base_lr,
          device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()