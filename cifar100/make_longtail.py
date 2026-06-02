"""
Tiny ImageNet → Long Tail Distribution 변환 스크립트

-균등한 데이터(class당 500개)를 Long tail distribution으로 변환.
-Head class는 많은 데이터, Tail class는 적은 데이터를 갖도록 설정.

Imbalance Factor (IF) = head class 수 / tail class 수
  ex) IF=100 → head class 500개, tail class 5개

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_longtail_counts(
    n_classes: int,
    max_samples: int,
    imbalance_factor: float,
) -> List[int]:
    """
    Exponential decay 방식으로 class별 샘플 수 계산.

    IF = max_samples / min_samples
    class i의 샘플 수 = max_samples * (1/IF)^(i / (n_classes-1))

    Returns: class별 샘플 수 리스트 (내림차순)
    """
    counts = []
    for i in range(n_classes):
        ratio = (1 / imbalance_factor) ** (i / (n_classes - 1))
        count = int(max_samples * ratio)
        count = max(count, 1)  # 최소 1개
        counts.append(count)
    return counts


def make_longtail_dataset(
    data_root: str,
    output_dir: str,
    imbalance_factor: float = 100,
    seed: int = 42,
):
    
    import datasets as hf_datasets

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드 
    logger.info("Loading Tiny ImageNet from: %s", data_root)
    hf_obj = hf_datasets.load_from_disk(data_root)

    if isinstance(hf_obj, hf_datasets.DatasetDict):
        train_ds = hf_obj["train"]
        valid_ds = (
            hf_obj.get("valid")
            or hf_obj.get("validation")
            or hf_obj.get("test")
        )
    else:
        raise ValueError("DatasetDict 형식이 아닙니다.")

    # 컬럼명 자동 감지 (label / fine_label 등)
    label_col = None
    for col in ["label", "fine_label", "coarse_label"]:
        if col in train_ds.features:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"label 컬럼을 찾을 수 없습니다.")
    logger.info("Label column: %s", label_col)

    label_feature = train_ds.features.get(label_col)
    class_names = getattr(label_feature, "names", None)
    n_classes = len(class_names) if class_names else int(max(train_ds[label_col])) + 1

    from collections import Counter
    label_counts = Counter(train_ds[label_col])
    max_samples = max(label_counts.values())

    logger.info("Classes: %d, Max samples per class: %d", n_classes, max_samples)
    logger.info("Imbalance Factor: %.0f", imbalance_factor)

    counts = compute_longtail_counts(n_classes, max_samples, imbalance_factor)

    logger.info("\nLong tail distribution:")
    logger.info("  Head class (class 0): %d samples", counts[0])
    logger.info("  Mid class (class %d): %d samples", n_classes // 2, counts[n_classes // 2])
    logger.info("  Tail class (class %d): %d samples", n_classes - 1, counts[-1])
    logger.info("  Total: %d samples (원본: %d)", sum(counts), max_samples * n_classes)

    logger.info("\nCollecting class indices...")
    class_to_indices: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx, label in enumerate(train_ds[label_col]):
        class_to_indices[int(label)].append(idx)

    # Long tail 샘플링 
    logger.info("Sampling long tail distribution...")
    selected_indices = []
    class_sample_counts = {}

    for class_idx, count in enumerate(counts):
        available = class_to_indices[class_idx]
        sampled = random.sample(available, min(count, len(available)))
        selected_indices.extend(sampled)
        class_sample_counts[class_idx] = len(sampled)

    # 인덱스 정렬
    selected_indices.sort()

    # 새 데이터셋 생성 
    logger.info("Creating long tail dataset (%d samples)...", len(selected_indices))
    longtail_train = train_ds.select(selected_indices)

    # 저장 
    logger.info("Saving to: %s", output_dir)
    longtail_dict = hf_datasets.DatasetDict({
        "train": longtail_train,
        "valid": valid_ds,
    })
    longtail_dict.save_to_disk(output_dir)

    # 통계 저장 
    stats = {
        "imbalance_factor": imbalance_factor,
        "n_classes": n_classes,
        "total_train_samples": len(selected_indices),
        "original_train_samples": max_samples * n_classes,
        "max_samples_per_class": counts[0],
        "min_samples_per_class": counts[-1],
        "class_distribution": {
            str(i): {
                "class_name": class_names[i] if class_names else str(i),
                "n_samples": class_sample_counts[i],
            }
            for i in range(n_classes)
        },
    }

    stats_path = os.path.join(output_dir, "longtail_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("\n===== Long Tail Dataset 생성 완료 =====")
    logger.info("저장 경로: %s", output_dir)
    logger.info("총 train 샘플: %d개 (원본 %d개)", len(selected_indices), max_samples * n_classes)
    logger.info("Head class (max): %d개", counts[0])
    logger.info("Tail class (min): %d개", counts[-1])
    logger.info("통계 저장: %s", stats_path)

    return stats


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset/tiny_imagenet",
        help="Tiny ImageNet HF save_to_disk 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset/tiny_imagenet_longtail",
        help="Long tail 데이터셋 저장 경로",
    )
    parser.add_argument(
        "--imbalance_factor",
        type=float,
        default=100,
        help="IF=100: head 500개, tail 5개 / IF=10: head 500개, tail 50개",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    make_longtail_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        imbalance_factor=args.imbalance_factor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()