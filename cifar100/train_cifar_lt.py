"""
CIFAR-100 Long Tail 학습 스크립트

"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# 전처리

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
])


# Dataset

class CIFARLTDataset(Dataset):
    """CIFAR-100 LT HF save_to_disk 형식 Dataset."""

    def __init__(self, hf_split, transform=None, label_col="fine_label"):
        self.ds = hf_split
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        image_obj = ex["img"]
        label = int(ex[self.label_col])

        if isinstance(image_obj, dict):
            if image_obj.get("bytes") is not None:
                image = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
            elif image_obj.get("path") is not None:
                image = Image.open(image_obj["path"]).convert("RGB")
            else:
                raise ValueError("Unsupported image dict format.")
        else:
            image = image_obj.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class PathDataset(Dataset):
    """
    (image_path, label) 리스트를 받아서 디스크에서 읽는 Dataset.
    Synthetic data용. OOM 방지를 위해 경로만 저장.
    """

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


# 데이터 로드 함수

def load_cifar_lt(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """CIFAR-100 LT 로드. train + valid 반환."""
    import datasets as hf_datasets

    logger.info("Loading CIFAR-100 LT from: %s", data_root)
    hf_obj = hf_datasets.load_from_disk(data_root)

    train_hf = hf_obj["train"]
    valid_hf = hf_obj.get("valid", hf_obj.get("validation", hf_obj.get("test")))

    label_feature = train_hf.features.get("fine_label")
    class_names = getattr(label_feature, "names", None)
    n_classes = len(class_names) if class_names else 100

    train_ds = CIFARLTDataset(train_hf, transform=TRAIN_TRANSFORM)
    valid_ds = CIFARLTDataset(valid_hf, transform=VAL_TRANSFORM)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # class별 샘플 수 확인 (Long tail 확인용)
    label_counts = Counter(train_hf["fine_label"])
    logger.info("Train: %d samples, %d classes", len(train_ds), n_classes)
    logger.info("Valid: %d samples", len(valid_ds))
    logger.info(
        "LT stats - Head: %d, Tail: %d",
        max(label_counts.values()),
        min(label_counts.values()),
    )

    return class_names, train_loader, valid_loader, label_counts


def _normalize_class_key(name: str) -> str:
    text = name.strip().lower().replace("_", " ")
    for ch in [",", ";", "/", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def load_synthetic_data(
    synth_data_root: str,
    class_names: List[str],
    tail_classes: Optional[List[int]] = None,
    use_clip_filter: bool = False,
    clip_threshold: float = 0.28,
    poor_class_threshold: float = 0.28,
    device: str = "cuda",
) -> List[Tuple[str, int]]:
    """
    Synthetic data 로드.
    tail_classes: tail class index 목록. 지정하면 해당 class만 로드.
    """
    from torchvision import datasets as tv_datasets

    root = Path(synth_data_root)
    train_root = root / "train" if (root / "train").is_dir() else root

    folder_ds = tv_datasets.ImageFolder(root=str(train_root))
    folder_idx_to_class = {v: k for k, v in folder_ds.class_to_idx.items()}

    # class name 매핑
    alias_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(class_names):
        raw = name.strip()
        for alias in [
            raw.lower(),
            raw.lower().replace("_", " "),
            _normalize_class_key(raw),
            raw.split(",")[0].strip().lower(),
            _normalize_class_key(raw.split(",")[0].strip()),
        ]:
            if alias:
                alias_to_idx[alias] = idx

    folder_to_val_idx: Dict[int, int] = {}
    for folder_idx, folder_name in folder_idx_to_class.items():
        mapped = None
        for cand in [
            folder_name.strip().lower(),
            folder_name.strip().lower().replace("_", " "),
            _normalize_class_key(folder_name),
            folder_name.split(",")[0].strip().lower(),
            _normalize_class_key(folder_name.split(",")[0].strip()),
        ]:
            mapped = alias_to_idx.get(cand)
            if mapped is not None:
                break
        if mapped is not None:
            folder_to_val_idx[folder_idx] = mapped

    # 경로만 수집
    samples: List[Tuple[str, int]] = []
    for img_path, folder_label in folder_ds.samples:
        val_label = folder_to_val_idx.get(folder_label)
        if val_label is None:
            continue
        # tail class만 쓸 경우 필터링
        if tail_classes is not None and val_label not in tail_classes:
            continue
        samples.append((img_path, val_label))

    logger.info("Synthetic samples loaded: %d", len(samples))

    # CLIP Filter 적용
    if use_clip_filter:
        from clip_filter import CLIPFilter
        logger.info("CLIPFilter 적용 중...")
        cf = CLIPFilter(device=device)
        samples, filter_stats = cf.filter(
            samples=samples,
            class_names=class_names,
            threshold=clip_threshold,
            poor_class_threshold=poor_class_threshold,
        )
        logger.info(
            "Filter 완료: %d → %d",
            filter_stats.total_before,
            filter_stats.total_after,
        )

    return samples


# 모델 

def build_resnet50(num_classes: int, device: str) -> nn.Module:
    """ResNet-50 생성 (scratch)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> nn.Module:
    """저장된 checkpoint 불러오기."""
    logger.info("Loading checkpoint: %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


# Validation 

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    class_names: Optional[List[str]] = None,
    label_counts: Optional[Counter] = None,
) -> Dict:
    """
    전체 Top-1 accuracy + head/tail class별 accuracy 계산.
    Long tail에서 tail class 성능 개선이 핵심이므로 분리해서 측정.
    """
    model.eval()
    correct = total = 0
    class_correct = Counter()
    class_total = Counter()

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        for p, l in zip(pred.cpu(), labels.cpu()):
            class_total[int(l)] += 1
            if p == l:
                class_correct[int(l)] += 1

    overall_acc = correct / total * 100

    # Head / Tail 분리 (label_counts 기준)
    head_correct = tail_correct = 0
    head_total = tail_total = 0
    if label_counts is not None:
        median = sorted(label_counts.values())[len(label_counts) // 2]
        for cls_idx in class_total:
            cnt = label_counts.get(cls_idx, 0)
            if cnt >= median:
                head_correct += class_correct[cls_idx]
                head_total += class_total[cls_idx]
            else:
                tail_correct += class_correct[cls_idx]
                tail_total += class_total[cls_idx]

    head_acc = head_correct / head_total * 100 if head_total > 0 else 0
    tail_acc = tail_correct / tail_total * 100 if tail_total > 0 else 0

    return {
        "overall": round(overall_acc, 2),
        "head": round(head_acc, 2),
        "tail": round(tail_acc, 2),
    }


# 학습 루프 

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 90,
    base_lr: float = 0.1,
    device: str = "cuda",
    output_dir: str = "./outputs",
    class_names: Optional[List[str]] = None,
    label_counts: Optional[Counter] = None,
):
    """ResNet-50 학습. SGD + CosineAnnealingLR."""
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr,
        momentum=0.9, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
    )

    log = {"epoch": [], "loss": [], "overall": [], "head": [], "tail": []}
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

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
        acc = validate(model, val_loader, device, class_names, label_counts)

        if acc["overall"] > best_acc:
            best_acc = acc["overall"]
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")

        log["epoch"].append(epoch)
        log["loss"].append(round(avg_loss, 4))
        log["overall"].append(acc["overall"])
        log["head"].append(acc["head"])
        log["tail"].append(acc["tail"])

        logger.info(
            "[Epoch %3d/%d] loss=%.4f | overall=%.2f%% | head=%.2f%% | tail=%.2f%% | best=%.2f%%",
            epoch, num_epochs, avg_loss,
            acc["overall"], acc["head"], acc["tail"], best_acc,
        )

    logger.info("Training complete. Best Overall: %.2f%%", best_acc)

    with open(f"{output_dir}/train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return model


# Main 

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        required=True,
        help="real: Real data로만 학습 / synthetic: Synthetic 추가 후 fine-tuning",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset/cifar100_longtail",
        help="CIFAR-100 LT 경로",
    )
    parser.add_argument(
        "--synth_data_root",
        type=str,
        default=None,
        help="Synthetic data 경로 (mode=synthetic일 때 필요)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="fine-tuning 시 불러올 checkpoint 경로",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/exp_cifar")
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--tail_only",
        action="store_true",
        help="Synthetic data를 tail class에만 추가할지 여부",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # CLIP Filter 관련
    parser.add_argument("--use_clip_filter", action="store_true")
    parser.add_argument("--clip_threshold", type=float, default=0.28)
    parser.add_argument("--poor_class_threshold", type=float, default=0.28)
    args = parser.parse_args()

    logger.info("Mode: %s | Device: %s", args.mode, args.device)

    # 데이터 로드
    class_names, train_loader, val_loader, label_counts = load_cifar_lt(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)

    # 모델 생성
    model = build_resnet50(num_classes=num_classes, device=args.device)

    # checkpoint 불러오기 (fine-tuning 시)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint, args.device)
        logger.info("Fine-tuning from checkpoint: %s", args.checkpoint)

    # Synthetic data 추가 (mode=synthetic일 때)
    if args.mode == "synthetic":
        if args.synth_data_root is None:
            raise ValueError("--synth_data_root 경로를 지정해주세요.")

        # tail class 목록 (label_counts 기준 하위 50%)
        tail_classes = None
        if args.tail_only:
            median = sorted(label_counts.values())[len(label_counts) // 2]
            tail_classes = [i for i, cnt in label_counts.items() if cnt < median]
            logger.info("Tail classes: %d개", len(tail_classes))

        synth_samples = load_synthetic_data(
            synth_data_root=args.synth_data_root,
            class_names=class_names,
            tail_classes=tail_classes,
            use_clip_filter=args.use_clip_filter,
            clip_threshold=args.clip_threshold,
            poor_class_threshold=args.poor_class_threshold,
            device=args.device,
        )

        synth_ds = PathDataset(synth_samples, transform=TRAIN_TRANSFORM)
        real_ds = train_loader.dataset

        # Real + Synthetic 합치기
        combined_ds = ConcatDataset([real_ds, synth_ds])
        train_loader = DataLoader(
            combined_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        logger.info(
            "Combined dataset: Real %d + Synthetic %d = %d",
            len(real_ds), len(synth_ds), len(combined_ds),
        )

    # 학습
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        base_lr=args.base_lr,
        device=args.device,
        output_dir=args.output_dir,
        class_names=class_names,
        label_counts=label_counts,
    )


if __name__ == "__main__":
    main()